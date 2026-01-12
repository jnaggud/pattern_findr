"""
News Sentiment Integration for Range Prediction

Uses Finnhub API for news sentiment data.
Finnhub free tier: 60 API calls/minute

Features:
- News sentiment scores
- News volume/buzz metrics
- Sentiment momentum (change over time)
- Big move prediction signals

API Docs: https://finnhub.io/docs/api/news-sentiment
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import sqlite3
from typing import Dict, Optional, List

# Cache database for news sentiment
NEWS_CACHE_DB = os.path.join(os.path.dirname(__file__), 'news_sentiment_cache.db')


class NewsSentimentManager:
    """
    Manages news sentiment data from Finnhub API.

    Free tier limits:
    - 60 API calls/minute
    - Basic news sentiment endpoint

    To get API key:
    1. Sign up at https://finnhub.io/register
    2. Copy API key from dashboard
    """

    def __init__(self, api_key: str = None):
        """
        Initialize news sentiment manager.

        Args:
            api_key: Finnhub API key. If None, tries to load from user_settings.json
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://finnhub.io/api/v1"

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.1  # ~60 calls/min = 1 per second + buffer

        # Cache
        self._init_cache_db()
        self._memory_cache = {}
        self._cache_duration = 3600  # 1 hour cache

    def _load_api_key(self) -> str:
        """Load Finnhub API key from user_settings.json"""
        try:
            settings_path = os.path.join(os.path.dirname(__file__), 'user_settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    return settings.get('finnhub_api_key', '')
        except Exception as e:
            print(f"Could not load Finnhub API key: {e}")
        return ''

    def _init_cache_db(self):
        """Initialize SQLite cache for news sentiment."""
        try:
            conn = sqlite3.connect(NEWS_CACHE_DB)
            cursor = conn.cursor()

            # News sentiment cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    buzz_articles_week REAL,
                    buzz_score REAL,
                    company_news_score REAL,
                    sector_avg_bullish REAL,
                    sector_avg_news_score REAL,
                    sentiment_bullish REAL,
                    sentiment_bearish REAL,
                    news_volume_24h INTEGER,
                    sentiment_score REAL,
                    sentiment_momentum REAL,
                    fetched_at TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # News articles cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    datetime TEXT,
                    headline TEXT,
                    source TEXT,
                    url TEXT,
                    sentiment REAL,
                    UNIQUE(symbol, headline, datetime)
                )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"News cache DB init error: {e}")

    def _rate_limit(self):
        """Respect Finnhub rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_news_sentiment(self, symbol: str) -> Dict:
        """
        Get news sentiment for a symbol from Finnhub.

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'SPY')

        Returns:
            Dict with sentiment data:
            - buzz_score: Measures news volume relative to baseline
            - sentiment_bullish: Percentage of bullish news (0-1)
            - sentiment_bearish: Percentage of bearish news (0-1)
            - company_news_score: Overall news sentiment score
            - sector_avg_bullish: Sector average bullish sentiment
        """
        if not self.api_key:
            return self._empty_sentiment()

        # Check memory cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if time.time() - cached['time'] < self._cache_duration:
                return cached['data']

        # Check database cache
        cached_data = self._get_cached_sentiment(symbol)
        if cached_data:
            self._memory_cache[cache_key] = {'data': cached_data, 'time': time.time()}
            return cached_data

        # Fetch from API
        self._rate_limit()

        try:
            url = f"{self.base_url}/news-sentiment"
            params = {'symbol': symbol, 'token': self.api_key}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Parse response
                sentiment_data = {
                    'buzz_articles_week': data.get('buzz', {}).get('articlesInLastWeek', 0),
                    'buzz_score': data.get('buzz', {}).get('buzz', 0),
                    'company_news_score': data.get('companyNewsScore', 0),
                    'sector_avg_bullish': data.get('sectorAverageBullishPercent', 0.5),
                    'sector_avg_news_score': data.get('sectorAverageNewsScore', 0),
                    'sentiment_bullish': data.get('sentiment', {}).get('bullishPercent', 0.5),
                    'sentiment_bearish': data.get('sentiment', {}).get('bearishPercent', 0.5),
                    'sentiment_score': data.get('sentiment', {}).get('bullishPercent', 0.5) -
                                       data.get('sentiment', {}).get('bearishPercent', 0.5),
                }

                # Save to cache
                self._save_sentiment_cache(symbol, sentiment_data)
                self._memory_cache[cache_key] = {'data': sentiment_data, 'time': time.time()}

                return sentiment_data

            elif response.status_code == 401:
                print(f"   Finnhub API key invalid or missing")
                return self._empty_sentiment()
            elif response.status_code == 429:
                print(f"   Finnhub rate limit exceeded")
                return self._empty_sentiment()
            else:
                print(f"   Finnhub API error: {response.status_code}")
                return self._empty_sentiment()

        except Exception as e:
            print(f"   News sentiment fetch error: {e}")
            return self._empty_sentiment()

    def get_company_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get recent company news articles.

        Args:
            symbol: Stock ticker
            days: Number of days of news to fetch

        Returns:
            List of news articles with headline, source, datetime
        """
        if not self.api_key:
            return []

        self._rate_limit()

        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date,
                'to': end_date,
                'token': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                articles = response.json()
                return articles[:50]  # Limit to 50 most recent
            else:
                return []

        except Exception as e:
            print(f"   Company news fetch error: {e}")
            return []

    def get_market_news(self, category: str = 'general') -> List[Dict]:
        """
        Get general market news.

        Args:
            category: 'general', 'forex', 'crypto', 'merger'

        Returns:
            List of market news articles
        """
        if not self.api_key:
            return []

        self._rate_limit()

        try:
            url = f"{self.base_url}/news"
            params = {'category': category, 'token': self.api_key}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()[:20]  # Limit to 20 most recent
            else:
                return []

        except Exception as e:
            print(f"   Market news fetch error: {e}")
            return []

    def _empty_sentiment(self) -> Dict:
        """Return empty sentiment data structure."""
        return {
            'buzz_articles_week': 0,
            'buzz_score': 0,
            'company_news_score': 0,
            'sector_avg_bullish': 0.5,
            'sector_avg_news_score': 0,
            'sentiment_bullish': 0.5,
            'sentiment_bearish': 0.5,
            'sentiment_score': 0,
        }

    def _get_cached_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get cached sentiment from database."""
        try:
            conn = sqlite3.connect(NEWS_CACHE_DB)
            cursor = conn.cursor()

            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute(
                '''SELECT * FROM news_sentiment WHERE symbol = ? AND date = ?''',
                (symbol, today)
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'buzz_articles_week': row[2],
                    'buzz_score': row[3],
                    'company_news_score': row[4],
                    'sector_avg_bullish': row[5],
                    'sector_avg_news_score': row[6],
                    'sentiment_bullish': row[7],
                    'sentiment_bearish': row[8],
                    'sentiment_score': row[10],
                }
            return None

        except Exception as e:
            return None

    def _save_sentiment_cache(self, symbol: str, data: Dict):
        """Save sentiment to database cache."""
        try:
            conn = sqlite3.connect(NEWS_CACHE_DB)
            cursor = conn.cursor()

            today = datetime.now().strftime('%Y-%m-%d')
            now = datetime.now().isoformat()

            cursor.execute('''
                INSERT OR REPLACE INTO news_sentiment
                (symbol, date, buzz_articles_week, buzz_score, company_news_score,
                 sector_avg_bullish, sector_avg_news_score, sentiment_bullish,
                 sentiment_bearish, news_volume_24h, sentiment_score, sentiment_momentum, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, today,
                data.get('buzz_articles_week', 0),
                data.get('buzz_score', 0),
                data.get('company_news_score', 0),
                data.get('sector_avg_bullish', 0.5),
                data.get('sector_avg_news_score', 0),
                data.get('sentiment_bullish', 0.5),
                data.get('sentiment_bearish', 0.5),
                0,  # news_volume_24h - would need to count articles
                data.get('sentiment_score', 0),
                0,  # sentiment_momentum - would need historical comparison
                now
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"   Sentiment cache save error: {e}")

    def get_historical_sentiment(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical sentiment data from cache.

        Args:
            symbol: Stock ticker
            days: Number of days of history

        Returns:
            DataFrame with historical sentiment indexed by date
        """
        try:
            conn = sqlite3.connect(NEWS_CACHE_DB)

            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            df = pd.read_sql_query(
                '''SELECT * FROM news_sentiment
                   WHERE symbol = ? AND date >= ?
                   ORDER BY date''',
                conn,
                params=(symbol, start_date)
            )

            conn.close()

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            return df

        except Exception as e:
            print(f"   Historical sentiment error: {e}")
            return pd.DataFrame()


def create_news_features(df: pd.DataFrame, ticker: str,
                         news_manager: NewsSentimentManager = None) -> pd.DataFrame:
    """
    Add news sentiment features to a DataFrame.

    This is the main integration function for price_prediction.py.

    Args:
        df: Price DataFrame with DatetimeIndex
        ticker: Stock ticker symbol
        news_manager: Optional pre-initialized NewsSentimentManager

    Returns:
        DataFrame with news features added
    """
    features = df.copy()

    # Initialize manager if not provided
    if news_manager is None:
        news_manager = NewsSentimentManager()

    # Get current sentiment
    sentiment = news_manager.get_news_sentiment(ticker)

    if sentiment['buzz_score'] > 0:  # Valid data
        print(f"   News sentiment: Bullish={sentiment['sentiment_bullish']:.2f}, "
              f"Bearish={sentiment['sentiment_bearish']:.2f}, "
              f"Buzz={sentiment['buzz_score']:.2f}")

        # Add point-in-time features (same value for all rows - will be forward-filled)
        features['news_sentiment_score'] = sentiment['sentiment_score']
        features['news_sentiment_bullish'] = sentiment['sentiment_bullish']
        features['news_sentiment_bearish'] = sentiment['sentiment_bearish']
        features['news_buzz_score'] = sentiment['buzz_score']
        features['news_company_score'] = sentiment['company_news_score']
        features['news_sector_avg_bullish'] = sentiment['sector_avg_bullish']

        # Derived features
        features['news_sentiment_vs_sector'] = (
            sentiment['sentiment_bullish'] - sentiment['sector_avg_bullish']
        )
        features['news_buzz_normalized'] = np.clip(sentiment['buzz_score'], 0, 3) / 3  # Normalize 0-1

        # Extreme sentiment flags
        features['news_extremely_bullish'] = float(sentiment['sentiment_bullish'] > 0.7)
        features['news_extremely_bearish'] = float(sentiment['sentiment_bearish'] > 0.7)

        # High buzz flag (lots of news = potential big move)
        features['news_high_buzz'] = float(sentiment['buzz_score'] > 1.5)

        # Combined big move indicator from news
        # High buzz + extreme sentiment = potential big move
        features['news_big_move_signal'] = float(
            sentiment['buzz_score'] > 1.5 and
            (sentiment['sentiment_bullish'] > 0.65 or sentiment['sentiment_bearish'] > 0.65)
        )

    else:
        # Add placeholder features when no data available
        print(f"   News sentiment: No data available for {ticker}")
        for col in ['news_sentiment_score', 'news_sentiment_bullish', 'news_sentiment_bearish',
                   'news_buzz_score', 'news_company_score', 'news_sector_avg_bullish',
                   'news_sentiment_vs_sector', 'news_buzz_normalized',
                   'news_extremely_bullish', 'news_extremely_bearish',
                   'news_high_buzz', 'news_big_move_signal']:
            features[col] = 0

    return features


# =============================================================================
# FREE NEWS SCRAPER (No API Key Required)
# Uses Google News RSS and simple sentiment analysis
# =============================================================================

# Sentiment word lists for financial news
BULLISH_WORDS = [
    'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb', 'boost', 'up',
    'bullish', 'positive', 'growth', 'profit', 'beat', 'exceed', 'record',
    'strong', 'upgrade', 'buy', 'optimistic', 'breakthrough', 'success',
    'momentum', 'outperform', 'higher', 'increase', 'expand', 'recovery'
]

BEARISH_WORDS = [
    'plunge', 'crash', 'fall', 'drop', 'decline', 'sink', 'tumble', 'down',
    'bearish', 'negative', 'loss', 'miss', 'below', 'weak', 'downgrade',
    'sell', 'pessimistic', 'risk', 'warning', 'concern', 'fear', 'trouble',
    'underperform', 'lower', 'decrease', 'contract', 'recession', 'layoff'
]


class FreeNewsScraper:
    """
    Free news scraper using Google News RSS - no API key required.

    Features:
    - Scrapes news from Google News RSS feeds
    - Simple word-based sentiment analysis
    - Counts news volume for buzz detection
    - Same interface as NewsSentimentManager

    Limitations:
    - Sentiment analysis is basic (word matching, not NLP)
    - May be rate limited by Google
    - Less accurate than Finnhub's ML-based sentiment
    """

    def __init__(self):
        """Initialize the free news scraper."""
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = 1800  # 30 minutes

    def get_news_sentiment(self, symbol: str) -> Dict:
        """
        Get news sentiment for a symbol by scraping Google News.

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'SPY')

        Returns:
            Dict with sentiment data (same structure as Finnhub)
        """
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H')}"
        if cache_key in self._cache:
            if time.time() < self._cache_expiry.get(cache_key, 0):
                return self._cache[cache_key]

        try:
            # Fetch news from Google News RSS
            articles = self._fetch_google_news(symbol)

            if not articles:
                return self._empty_sentiment()

            # Analyze sentiment
            sentiment_scores = []
            for article in articles:
                score = self._analyze_headline(article['title'])
                sentiment_scores.append(score)

            # Calculate aggregate metrics
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            bullish_count = sum(1 for s in sentiment_scores if s > 0.1)
            bearish_count = sum(1 for s in sentiment_scores if s < -0.1)
            total_with_sentiment = bullish_count + bearish_count

            sentiment_data = {
                'buzz_articles_week': len(articles),
                'buzz_score': min(len(articles) / 10, 3),  # Normalize to 0-3 range
                'company_news_score': avg_sentiment,
                'sector_avg_bullish': 0.5,  # Default (no sector comparison)
                'sector_avg_news_score': 0,
                'sentiment_bullish': bullish_count / max(total_with_sentiment, 1),
                'sentiment_bearish': bearish_count / max(total_with_sentiment, 1),
                'sentiment_score': avg_sentiment,
            }

            # Cache result
            self._cache[cache_key] = sentiment_data
            self._cache_expiry[cache_key] = time.time() + self._cache_duration

            return sentiment_data

        except Exception as e:
            print(f"   Free scraper error: {e}")
            return self._empty_sentiment()

    def _fetch_google_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """Fetch news articles from Google News RSS."""
        import xml.etree.ElementTree as ET

        try:
            # Google News RSS URL for stock ticker
            # Using stock news query format
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return []

            # Parse RSS XML
            root = ET.fromstring(response.content)

            articles = []
            for item in root.findall('.//item')[:max_articles]:
                title = item.find('title')
                pub_date = item.find('pubDate')
                link = item.find('link')

                if title is not None:
                    articles.append({
                        'title': title.text or '',
                        'pubDate': pub_date.text if pub_date is not None else '',
                        'link': link.text if link is not None else ''
                    })

            return articles

        except Exception as e:
            print(f"   Google News fetch error: {e}")
            return []

    def _analyze_headline(self, headline: str) -> float:
        """
        Analyze sentiment of a news headline.

        Returns:
            Float from -1 (very bearish) to +1 (very bullish)
        """
        if not headline:
            return 0

        headline_lower = headline.lower()

        bullish_count = sum(1 for word in BULLISH_WORDS if word in headline_lower)
        bearish_count = sum(1 for word in BEARISH_WORDS if word in headline_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0

        return (bullish_count - bearish_count) / total

    def _empty_sentiment(self) -> Dict:
        """Return empty sentiment data structure."""
        return {
            'buzz_articles_week': 0,
            'buzz_score': 0,
            'company_news_score': 0,
            'sector_avg_bullish': 0.5,
            'sector_avg_news_score': 0,
            'sentiment_bullish': 0.5,
            'sentiment_bearish': 0.5,
            'sentiment_score': 0,
        }

    def get_company_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get recent company news articles."""
        return self._fetch_google_news(symbol, max_articles=50)


def get_news_manager(prefer_api: bool = True) -> 'NewsSentimentManager':
    """
    Get the best available news sentiment manager.

    Args:
        prefer_api: If True, prefer Finnhub API over free scraper

    Returns:
        NewsSentimentManager if API key available, else FreeNewsScraper
    """
    if prefer_api:
        manager = NewsSentimentManager()
        if manager.api_key:
            return manager

    # Fallback to free scraper
    print("   Using free news scraper (no API key)")
    return FreeNewsScraper()


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("NEWS SENTIMENT TEST")
    print("=" * 60)

    # Test Finnhub (if API key available)
    manager = NewsSentimentManager()

    if not manager.api_key:
        print("\nNo Finnhub API key found.")
        print("To use Finnhub API:")
        print("1. Sign up at https://finnhub.io/register")
        print("2. Add to user_settings.json: {\"finnhub_api_key\": \"YOUR_KEY\"}")
    else:
        print(f"\nFinnhub API key found: {manager.api_key[:8]}...")

        # Test SPY sentiment
        print("\n--- Testing Finnhub SPY Sentiment ---")
        sentiment = manager.get_news_sentiment('SPY')
        for k, v in sentiment.items():
            print(f"  {k}: {v}")

    # Test Free Scraper (always available)
    print("\n" + "=" * 60)
    print("TESTING FREE NEWS SCRAPER (No API Key Required)")
    print("=" * 60)

    scraper = FreeNewsScraper()

    # Test SPY
    print("\n--- Free Scraper: SPY Sentiment ---")
    sentiment = scraper.get_news_sentiment('SPY')
    for k, v in sentiment.items():
        print(f"  {k}: {v}")

    # Test AAPL
    print("\n--- Free Scraper: AAPL Sentiment ---")
    sentiment = scraper.get_news_sentiment('AAPL')
    for k, v in sentiment.items():
        print(f"  {k}: {v}")

    # Get sample headlines
    print("\n--- Sample Headlines (AAPL) ---")
    articles = scraper.get_company_news('AAPL')
    for article in articles[:5]:
        score = scraper._analyze_headline(article['title'])
        sentiment_label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        print(f"  [{sentiment_label:^8}] {article['title'][:70]}...")

    # Test get_news_manager helper
    print("\n" + "=" * 60)
    print("TESTING AUTO-SELECT (get_news_manager)")
    print("=" * 60)

    auto_manager = get_news_manager(prefer_api=True)
    print(f"\nSelected manager type: {type(auto_manager).__name__}")
    sentiment = auto_manager.get_news_sentiment('NVDA')
    print(f"NVDA sentiment score: {sentiment['sentiment_score']:.2f}")
    print(f"NVDA buzz score: {sentiment['buzz_score']:.2f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
