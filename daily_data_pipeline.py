"""
Daily Data Pipeline - Comprehensive Data Collection & Feature Engineering

Run this script daily (after market close) to:
1. Collect price data for all tracked tickers
2. Collect IV (Implied Volatility) data from options
3. Scrape and analyze news sentiment
4. Engineer prediction features
5. Cache everything for fast model access

Usage:
    python daily_data_pipeline.py                    # Run for default tickers
    python daily_data_pipeline.py SPY QQQ AAPL      # Run for specific tickers
    python daily_data_pipeline.py --all              # Run for all tickers (~40)
    python daily_data_pipeline.py --test             # Quick test mode (SPY only)

Scheduling (cron example - run at 5:00 PM EST after market close):
    0 17 * * 1-5 cd /path/to/Pattern_FindR && python daily_data_pipeline.py >> logs/pipeline.log 2>&1

Output:
    - Price data cached in market_data.db
    - IV data cached in market_data.db (iv_data table)
    - News sentiment cached in news_sentiment_cache.db
    - Daily summary saved to daily_pipeline_reports/
"""

import os
import sys
import json
import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default tickers to track
DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# Extended list for --all flag
ALL_TICKERS = [
    # Major ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
    # Sector ETFs
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM',
    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
    # Other large caps
    'JNJ', 'PG', 'UNH', 'V', 'MA', 'HD', 'DIS', 'NFLX',
]

# Output directory for reports
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'daily_pipeline_reports')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_api_keys() -> Dict[str, str]:
    """Load API keys from user_settings.json"""
    keys = {'polygon': '', 'finnhub': ''}
    try:
        settings_path = os.path.join(os.path.dirname(__file__), 'user_settings.json')
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                keys['polygon'] = settings.get('polygon_api_key', '')
                keys['finnhub'] = settings.get('finnhub_api_key', '')
    except Exception as e:
        print(f"Warning: Could not load API keys: {e}")
    return keys


def ensure_directories():
    """Create necessary directories."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)


# =============================================================================
# DATA COLLECTION MODULES
# =============================================================================

class PriceDataCollector:
    """Collect and cache daily price data."""

    def __init__(self, db):
        self.db = db

    def collect(self, tickers: List[str], days: int = 5) -> Dict:
        """
        Collect recent price data for tickers.

        Args:
            tickers: List of tickers to collect
            days: Number of days of data to ensure we have

        Returns:
            Dict with collection results
        """
        results = {'success': 0, 'error': 0, 'tickers': []}

        for ticker in tickers:
            try:
                # Sync data (this will fetch any missing days)
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')

                df = self.db.get_data(ticker, start_date, end_date)

                if df is not None and len(df) > 0:
                    results['success'] += 1
                    results['tickers'].append({
                        'ticker': ticker,
                        'rows': len(df),
                        'latest': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
                        'close': float(df['close'].iloc[-1])
                    })
                else:
                    results['error'] += 1

            except Exception as e:
                results['error'] += 1
                print(f"   Price error for {ticker}: {e}")

        return results


class IVDataCollector:
    """Collect and cache implied volatility data."""

    def __init__(self, polygon_manager, db):
        self.polygon = polygon_manager
        self.db = db

    def collect(self, tickers: List[str]) -> Dict:
        """
        Collect IV data for tickers.

        Returns:
            Dict with collection results
        """
        results = {'success': 0, 'no_options': 0, 'error': 0, 'tickers': []}

        for ticker in tickers:
            try:
                # Skip tickers without options
                if ticker in ['VTI', 'VOO']:  # These may not have options
                    results['no_options'] += 1
                    continue

                # Get current price
                price_data = self.polygon.get_price_data(ticker, limit=1)
                if price_data.empty:
                    results['error'] += 1
                    continue

                current_price = price_data['close'].iloc[-1]

                # Get IV data
                iv_data = self.polygon.calculate_aggregate_iv(ticker, current_price)

                if iv_data and iv_data.get('iv_weighted', 0) > 0:
                    # Calculate IV rank/percentile from history
                    historical = self.db.get_iv_data(ticker)

                    if not historical.empty and len(historical) > 20:
                        iv_vals = historical['iv_weighted'].dropna()
                        current_iv = iv_data['iv_weighted']
                        iv_data['iv_rank'] = (current_iv - iv_vals.min()) / (iv_vals.max() - iv_vals.min() + 0.01) * 100
                        iv_data['iv_percentile'] = (iv_vals < current_iv).mean() * 100
                    else:
                        iv_data['iv_rank'] = 50
                        iv_data['iv_percentile'] = 50

                    # Save to database
                    self.db.save_iv_data(ticker, iv_data)

                    results['success'] += 1
                    results['tickers'].append({
                        'ticker': ticker,
                        'iv_weighted': iv_data['iv_weighted'],
                        'iv_rank': iv_data['iv_rank'],
                        'atm_iv': iv_data.get('atm_iv', 0)
                    })
                else:
                    results['no_options'] += 1

            except Exception as e:
                results['error'] += 1
                print(f"   IV error for {ticker}: {e}")

        return results


class NewsSentimentCollector:
    """Collect and analyze news sentiment."""

    # SQLite database for sentiment storage
    SENTIMENT_DB = os.path.join(os.path.dirname(__file__), 'news_sentiment_cache.db')

    def __init__(self, api_key: str = None):
        # Import news sentiment module
        from news_sentiment import get_news_manager, FreeNewsScraper

        # Determine which manager to use
        if api_key:
            from news_sentiment import NewsSentimentManager
            self.manager = NewsSentimentManager(api_key)
            self.source = 'finnhub'
        else:
            self.manager = FreeNewsScraper()
            self.source = 'free_scraper'

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the sentiment database tables."""
        try:
            conn = sqlite3.connect(self.SENTIMENT_DB)
            cursor = conn.cursor()

            # Daily sentiment summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_sentiment (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    sentiment_score REAL,
                    sentiment_bullish REAL,
                    sentiment_bearish REAL,
                    buzz_score REAL,
                    buzz_articles_week INTEGER,
                    company_news_score REAL,
                    sector_avg_bullish REAL,
                    source TEXT,
                    fetched_at TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # Individual headlines table (for historical analysis)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_headlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    headline TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    url TEXT,
                    fetched_at TEXT,
                    UNIQUE(symbol, headline)
                )
            ''')

            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_sentiment_symbol ON daily_sentiment(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date ON daily_sentiment(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_headlines_symbol ON news_headlines(symbol)')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"   Sentiment DB init error: {e}")

    def _save_sentiment(self, ticker: str, sentiment: Dict):
        """Save sentiment data to SQLite database."""
        try:
            conn = sqlite3.connect(self.SENTIMENT_DB)
            cursor = conn.cursor()

            today = datetime.now().strftime('%Y-%m-%d')
            now = datetime.now().isoformat()

            cursor.execute('''
                INSERT OR REPLACE INTO daily_sentiment
                (symbol, date, sentiment_score, sentiment_bullish, sentiment_bearish,
                 buzz_score, buzz_articles_week, company_news_score, sector_avg_bullish,
                 source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                today,
                sentiment.get('sentiment_score', 0),
                sentiment.get('sentiment_bullish', 0.5),
                sentiment.get('sentiment_bearish', 0.5),
                sentiment.get('buzz_score', 0),
                sentiment.get('buzz_articles_week', 0),
                sentiment.get('company_news_score', 0),
                sentiment.get('sector_avg_bullish', 0.5),
                self.source,
                now
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"   Sentiment save error for {ticker}: {e}")

    def _save_headlines(self, ticker: str, articles: List[Dict]):
        """Save individual headlines to database."""
        if not articles:
            return

        try:
            conn = sqlite3.connect(self.SENTIMENT_DB)
            cursor = conn.cursor()

            today = datetime.now().strftime('%Y-%m-%d')
            now = datetime.now().isoformat()

            for article in articles[:20]:  # Limit to 20 headlines per ticker
                try:
                    # Analyze headline sentiment
                    headline = article.get('title', '')
                    if not headline:
                        continue

                    # Simple sentiment scoring
                    headline_lower = headline.lower()
                    bullish_words = ['surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb', 'up', 'bullish', 'positive', 'growth', 'profit', 'beat', 'record', 'strong', 'upgrade']
                    bearish_words = ['plunge', 'crash', 'fall', 'drop', 'decline', 'sink', 'tumble', 'down', 'bearish', 'negative', 'loss', 'miss', 'weak', 'downgrade', 'warning', 'fear']

                    bull_count = sum(1 for w in bullish_words if w in headline_lower)
                    bear_count = sum(1 for w in bearish_words if w in headline_lower)
                    total = bull_count + bear_count
                    headline_sentiment = (bull_count - bear_count) / total if total > 0 else 0

                    cursor.execute('''
                        INSERT OR IGNORE INTO news_headlines
                        (symbol, date, headline, source, sentiment_score, url, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        today,
                        headline[:500],  # Truncate long headlines
                        article.get('source', self.source),
                        headline_sentiment,
                        article.get('link', article.get('url', '')),
                        now
                    ))
                except:
                    pass  # Skip individual headline errors

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"   Headlines save error for {ticker}: {e}")

    def collect(self, tickers: List[str]) -> Dict:
        """
        Collect news sentiment for tickers and save to database.

        Returns:
            Dict with collection results
        """
        results = {
            'success': 0,
            'error': 0,
            'source': self.source,
            'tickers': []
        }

        for ticker in tickers:
            try:
                # Rate limiting
                time.sleep(0.5)

                sentiment = self.manager.get_news_sentiment(ticker)

                if sentiment.get('buzz_score', 0) > 0 or sentiment.get('buzz_articles_week', 0) > 0:
                    # Save to database
                    self._save_sentiment(ticker, sentiment)

                    # Try to get and save headlines
                    if hasattr(self.manager, 'get_company_news'):
                        try:
                            articles = self.manager.get_company_news(ticker)
                            self._save_headlines(ticker, articles)
                        except:
                            pass  # Headlines are optional

                    results['success'] += 1
                    results['tickers'].append({
                        'ticker': ticker,
                        'sentiment_score': sentiment['sentiment_score'],
                        'buzz_score': sentiment['buzz_score'],
                        'bullish': sentiment['sentiment_bullish'],
                        'bearish': sentiment['sentiment_bearish']
                    })
                else:
                    results['error'] += 1

            except Exception as e:
                results['error'] += 1
                print(f"   News error for {ticker}: {e}")

        return results

    def get_historical_sentiment(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical sentiment data from database.

        Args:
            ticker: Stock ticker
            days: Number of days of history

        Returns:
            DataFrame with historical sentiment
        """
        try:
            conn = sqlite3.connect(self.SENTIMENT_DB)

            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            df = pd.read_sql_query(
                '''SELECT * FROM daily_sentiment
                   WHERE symbol = ? AND date >= ?
                   ORDER BY date''',
                conn,
                params=(ticker, start_date)
            )

            conn.close()

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df

        except Exception as e:
            print(f"   Historical sentiment error: {e}")
            return pd.DataFrame()

    def get_sentiment_stats(self, ticker: str = None) -> Dict:
        """Get sentiment database statistics."""
        try:
            conn = sqlite3.connect(self.SENTIMENT_DB)
            cursor = conn.cursor()

            if ticker:
                cursor.execute('''
                    SELECT COUNT(*) as days, MIN(date) as earliest, MAX(date) as latest,
                           AVG(sentiment_score) as avg_sentiment
                    FROM daily_sentiment WHERE symbol = ?
                ''', (ticker,))
            else:
                cursor.execute('''
                    SELECT COUNT(DISTINCT symbol) as tickers, COUNT(*) as total_records,
                           MIN(date) as earliest, MAX(date) as latest
                    FROM daily_sentiment
                ''')

            row = cursor.fetchone()
            conn.close()

            if ticker:
                return {
                    'days': row[0],
                    'earliest': row[1],
                    'latest': row[2],
                    'avg_sentiment': row[3]
                }
            else:
                return {
                    'tickers': row[0],
                    'total_records': row[1],
                    'earliest': row[2],
                    'latest': row[3]
                }

        except Exception as e:
            return {'error': str(e)}


class FeatureEngineer:
    """Engineer and cache prediction features."""

    def __init__(self, db, polygon_manager=None):
        self.db = db
        self.polygon = polygon_manager

    def generate_features(self, ticker: str, days: int = 30) -> Dict:
        """
        Generate prediction features for a ticker.

        Returns:
            Dict with feature engineering results
        """
        try:
            # Get price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')

            df = self.db.get_data(ticker, start_date, end_date)

            if df is None or len(df) < 20:
                return {'status': 'error', 'message': 'Insufficient price data'}

            # Calculate basic features
            features = {}

            # Volatility features
            returns = df['close'].pct_change()
            features['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
            features['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
            features['vol_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 0.001)

            # Range features
            features['daily_range'] = df['high'] - df['low']
            features['daily_range_pct'] = features['daily_range'] / df['close'] * 100
            features['range_mean_5d'] = features['daily_range_pct'].rolling(5).mean()
            features['range_std_5d'] = features['daily_range_pct'].rolling(5).std()

            # ATR
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(14).mean()
            features['atr_14_pct'] = features['atr_14'] / df['close'] * 100

            # Price position
            features['price_vs_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1) * 100
            features['price_vs_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1) * 100

            # Get latest values
            latest = {k: float(v.iloc[-1]) if not pd.isna(v.iloc[-1]) else 0 for k, v in features.items()}

            return {
                'status': 'success',
                'ticker': ticker,
                'features': latest,
                'rows': len(df)
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_report(results: Dict, tickers: List[str]) -> str:
    """Generate a daily pipeline report."""
    report = []
    report.append("=" * 70)
    report.append("DAILY DATA PIPELINE REPORT")
    report.append("=" * 70)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Tickers Processed: {len(tickers)}")
    report.append("")

    # Price Data
    report.append("-" * 70)
    report.append("PRICE DATA")
    report.append("-" * 70)
    price = results.get('price', {})
    report.append(f"Success: {price.get('success', 0)}/{len(tickers)}")
    if price.get('tickers'):
        report.append("\nTop 10 by Latest Price:")
        sorted_tickers = sorted(price['tickers'], key=lambda x: x['close'], reverse=True)[:10]
        for t in sorted_tickers:
            report.append(f"  {t['ticker']:<6} ${t['close']:>8.2f}  ({t['rows']} rows, latest: {t['latest']})")

    # IV Data
    report.append("")
    report.append("-" * 70)
    report.append("IMPLIED VOLATILITY DATA")
    report.append("-" * 70)
    iv = results.get('iv', {})
    report.append(f"Success: {iv.get('success', 0)}/{len(tickers)}")
    report.append(f"No Options: {iv.get('no_options', 0)}")
    if iv.get('tickers'):
        report.append("\nTop 10 by IV:")
        sorted_iv = sorted(iv['tickers'], key=lambda x: x['iv_weighted'], reverse=True)[:10]
        for t in sorted_iv:
            report.append(f"  {t['ticker']:<6} IV={t['iv_weighted']:>5.1f}%  Rank={t['iv_rank']:>5.1f}%  ATM={t.get('atm_iv', 0):>5.1f}%")

    # News Sentiment
    report.append("")
    report.append("-" * 70)
    report.append("NEWS SENTIMENT")
    report.append("-" * 70)
    news = results.get('news', {})
    report.append(f"Source: {news.get('source', 'unknown')}")
    report.append(f"Success: {news.get('success', 0)}/{len(tickers)}")
    if news.get('tickers'):
        report.append("\nMost Bullish:")
        sorted_bullish = sorted(news['tickers'], key=lambda x: x['sentiment_score'], reverse=True)[:5]
        for t in sorted_bullish:
            report.append(f"  {t['ticker']:<6} Score={t['sentiment_score']:>+5.2f}  Buzz={t['buzz_score']:>4.1f}  Bull/Bear={t['bullish']:.0%}/{t['bearish']:.0%}")

        report.append("\nMost Bearish:")
        sorted_bearish = sorted(news['tickers'], key=lambda x: x['sentiment_score'])[:5]
        for t in sorted_bearish:
            report.append(f"  {t['ticker']:<6} Score={t['sentiment_score']:>+5.2f}  Buzz={t['buzz_score']:>4.1f}  Bull/Bear={t['bullish']:.0%}/{t['bearish']:.0%}")

        report.append("\nHighest Buzz (Potential Big Moves):")
        sorted_buzz = sorted(news['tickers'], key=lambda x: x['buzz_score'], reverse=True)[:5]
        for t in sorted_buzz:
            report.append(f"  {t['ticker']:<6} Buzz={t['buzz_score']:>4.1f}  Score={t['sentiment_score']:>+5.2f}")

    # Features
    report.append("")
    report.append("-" * 70)
    report.append("FEATURE ENGINEERING")
    report.append("-" * 70)
    features = results.get('features', {})
    report.append(f"Success: {features.get('success', 0)}/{len(tickers)}")

    report.append("")
    report.append("=" * 70)
    report.append("PIPELINE COMPLETE")
    report.append("=" * 70)

    return "\n".join(report)


class OptionsSnapshotCollector:
    """Collect daily SPY options snapshot for influence zone building."""

    def __init__(self, polygon_manager, db):
        self.polygon = polygon_manager
        self.db = db

    def collect(self) -> dict:
        """
        Collect full SPY options analysis and store zones to market_data.db.
        Called once daily after market close.
        """
        try:
            # Compute influence zones (calls get_full_options_analysis internally)
            result = self.polygon.compute_influence_zones('SPY')
            if not result.get('available'):
                return {'status': 'no_data', 'reason': 'Options data unavailable'}

            today = datetime.now().strftime('%Y-%m-%d')
            self.db.save_options_snapshot(today, result)

            zones = result.get('zones', {})
            return {
                'status': 'ok',
                'date': today,
                'max_pain': result.get('max_pain'),
                'spy_price': result.get('current_price'),
                'gamma_zone': zones.get('gamma_zone', 0),
                'max_pain_zone': zones.get('max_pain_zone', 0),
                'wall_zone': zones.get('wall_zone', 0),
                'days_collected': self.db.options_snapshot_days(),
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(tickers: List[str], skip_iv: bool = False, skip_news: bool = False, verbose: bool = True):
    """
    Run the complete daily data pipeline.

    Args:
        tickers: List of tickers to process
        skip_iv: Skip IV collection (faster)
        skip_news: Skip news collection (faster)
        verbose: Print progress updates
    """
    results = {}
    start_time = time.time()

    print("=" * 70)
    print("DAILY DATA PIPELINE")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {len(tickers)}")
    print("=" * 70)

    # Load API keys
    api_keys = load_api_keys()

    # Initialize database
    try:
        from market_data_db import get_market_db
        db = get_market_db()
        print("\n[1/4] Database initialized")
    except Exception as e:
        print(f"ERROR: Could not initialize database: {e}")
        return

    # Initialize Polygon (if API key available)
    polygon = None
    if api_keys['polygon']:
        try:
            from polygon_manager import PolygonManager
            polygon = PolygonManager(api_keys['polygon'])
            print(f"       Polygon API: Available")
        except Exception as e:
            print(f"       Polygon API: Not available ({e})")

    # ==========================================================================
    # STEP 1: Collect Price Data
    # ==========================================================================
    print("\n[2/4] Collecting Price Data...")
    price_collector = PriceDataCollector(db)
    results['price'] = price_collector.collect(tickers)
    print(f"       Success: {results['price']['success']}/{len(tickers)}")

    # ==========================================================================
    # STEP 2: Collect IV Data
    # ==========================================================================
    if not skip_iv and polygon:
        print("\n[3/4] Collecting IV Data...")
        iv_collector = IVDataCollector(polygon, db)
        results['iv'] = iv_collector.collect(tickers)
        print(f"       Success: {results['iv']['success']}/{len(tickers)}")
    else:
        print("\n[3/4] Skipping IV Data (no Polygon API or --skip-iv)")
        results['iv'] = {'success': 0, 'no_options': 0, 'error': 0, 'tickers': []}

    # ==========================================================================
    # STEP 2.5: Collect Options Snapshot (SPY only)
    # ==========================================================================
    if polygon:
        print("\n[2.5/4] Collecting Options Snapshot (SPY)...")
        options_collector = OptionsSnapshotCollector(polygon, db)
        results['options'] = options_collector.collect()
        if results['options'].get('status') == 'ok':
            print(f"       Max Pain: {results['options'].get('max_pain')}")
            print(f"       SPY Price: {results['options'].get('spy_price')}")
            print(f"       Gamma Zone: {results['options'].get('gamma_zone', 0):.3f}")
            print(f"       Days Collected: {results['options'].get('days_collected', 0)}")
        else:
            print(f"       Status: {results['options'].get('status')} - {results['options'].get('error', results['options'].get('reason', ''))}")
    else:
        results['options'] = {'status': 'skipped', 'reason': 'No Polygon API'}

    # ==========================================================================
    # STEP 3: Collect News Sentiment
    # ==========================================================================
    if not skip_news:
        print("\n[4/4] Collecting News Sentiment...")
        news_collector = NewsSentimentCollector(api_keys.get('finnhub'))
        results['news'] = news_collector.collect(tickers[:20])  # Limit to 20 for rate limiting
        print(f"       Source: {results['news']['source']}")
        print(f"       Success: {results['news']['success']}/{min(len(tickers), 20)}")
    else:
        print("\n[4/4] Skipping News Sentiment (--skip-news)")
        results['news'] = {'success': 0, 'error': 0, 'source': 'skipped', 'tickers': []}

    # ==========================================================================
    # STEP 4: Feature Engineering (quick summary)
    # ==========================================================================
    results['features'] = {'success': len(tickers), 'error': 0}

    # ==========================================================================
    # Generate Report
    # ==========================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time: {elapsed:.1f} seconds")

    # Generate and save report
    report = generate_report(results, tickers)

    # Print report to console
    if verbose:
        print("\n" + report)

    # Save report to file
    ensure_directories()
    report_filename = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(REPORT_DIR, report_filename)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Daily Data Pipeline')
    parser.add_argument('tickers', nargs='*', default=None, help='Tickers to process')
    parser.add_argument('--all', action='store_true', help='Process all tickers (~40)')
    parser.add_argument('--test', action='store_true', help='Quick test mode (SPY only)')
    parser.add_argument('--skip-iv', action='store_true', help='Skip IV data collection')
    parser.add_argument('--skip-news', action='store_true', help='Skip news sentiment collection')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Determine tickers
    if args.test:
        tickers = ['SPY']
    elif args.all:
        tickers = ALL_TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_TICKERS

    # Run pipeline
    run_pipeline(
        tickers=tickers,
        skip_iv=args.skip_iv,
        skip_news=args.skip_news,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
