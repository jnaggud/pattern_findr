"""
Historical News Backfill Script

Collects historical news data from multiple sources to backfill the sentiment database.
This allows testing how well news sentiment helps the prediction model.

Sources:
1. Wayback Machine (archive.org) - Historical snapshots of news sites
2. NewsAPI.org - 30 days of historical news (free tier)
3. GDELT Project - Free historical news database
4. Polygon.io - Historical news (if you have API key)

Usage:
    python backfill_historical_news.py SPY --days 90
    python backfill_historical_news.py SPY AAPL NVDA --days 60
    python backfill_historical_news.py --all --days 30

Output:
    - News sentiment saved to news_sentiment_cache.db
    - Headlines saved with sentiment scores
    - Summary report generated
"""

import os
import sys
import json
import sqlite3
import argparse
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Database path
NEWS_DB = os.path.join(os.path.dirname(__file__), 'news_sentiment_cache.db')

# Sentiment word lists
BULLISH_WORDS = [
    'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb', 'boost', 'up',
    'bullish', 'positive', 'growth', 'profit', 'beat', 'exceed', 'record',
    'strong', 'upgrade', 'buy', 'optimistic', 'breakthrough', 'success',
    'momentum', 'outperform', 'higher', 'increase', 'expand', 'recovery',
    'bull', 'boom', 'advance', 'improve', 'accelerate', 'peak'
]

BEARISH_WORDS = [
    'plunge', 'crash', 'fall', 'drop', 'decline', 'sink', 'tumble', 'down',
    'bearish', 'negative', 'loss', 'miss', 'below', 'weak', 'downgrade',
    'sell', 'pessimistic', 'risk', 'warning', 'concern', 'fear', 'trouble',
    'underperform', 'lower', 'decrease', 'contract', 'recession', 'layoff',
    'bear', 'bust', 'slump', 'worsen', 'decelerate', 'bottom', 'crisis'
]

DEFAULT_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


class HistoricalNewsBackfill:
    """Backfill historical news data from multiple sources."""

    def __init__(self):
        self._init_db()
        self._load_api_keys()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()

        # Ensure tables exist
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

        # Historical backfill tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backfill_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                source TEXT,
                articles_found INTEGER,
                status TEXT,
                created_at TEXT,
                UNIQUE(symbol, date, source)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_api_keys(self):
        """Load API keys from user_settings.json."""
        self.api_keys = {}
        try:
            settings_path = os.path.join(os.path.dirname(__file__), 'user_settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    self.api_keys = {
                        'newsapi': settings.get('newsapi_key', ''),
                        'polygon': settings.get('polygon_api_key', ''),
                    }
        except:
            pass

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using word matching."""
        if not text:
            return 0

        text_lower = text.lower()
        bull_count = sum(1 for word in BULLISH_WORDS if word in text_lower)
        bear_count = sum(1 for word in BEARISH_WORDS if word in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return 0

        return (bull_count - bear_count) / total

    def save_daily_sentiment(self, symbol: str, date: str, headlines: List[Dict], source: str):
        """Calculate and save daily sentiment from headlines."""
        if not headlines:
            return

        # Calculate sentiment scores
        sentiments = [self.analyze_sentiment(h.get('title', '')) for h in headlines]
        valid_sentiments = [s for s in sentiments if s != 0]

        if not valid_sentiments:
            avg_sentiment = 0
            bullish_pct = 0.5
            bearish_pct = 0.5
        else:
            avg_sentiment = np.mean(valid_sentiments)
            bullish_pct = sum(1 for s in valid_sentiments if s > 0.1) / len(valid_sentiments)
            bearish_pct = sum(1 for s in valid_sentiments if s < -0.1) / len(valid_sentiments)

        buzz_score = min(len(headlines) / 10, 3)  # Normalize to 0-3

        # Save to database
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_sentiment
                (symbol, date, sentiment_score, sentiment_bullish, sentiment_bearish,
                 buzz_score, buzz_articles_week, company_news_score, sector_avg_bullish,
                 source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, date, avg_sentiment, bullish_pct, bearish_pct,
                buzz_score, len(headlines), avg_sentiment, 0.5,
                source, datetime.now().isoformat()
            ))

            # Save individual headlines
            for h in headlines[:30]:  # Limit to 30 per day
                title = h.get('title', '')[:500]
                if title:
                    cursor.execute('''
                        INSERT OR IGNORE INTO news_headlines
                        (symbol, date, headline, source, sentiment_score, url, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, date, title, source,
                        self.analyze_sentiment(title),
                        h.get('url', ''),
                        datetime.now().isoformat()
                    ))

            # Log the backfill
            cursor.execute('''
                INSERT OR REPLACE INTO backfill_log
                (symbol, date, source, articles_found, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, date, source, len(headlines), 'success', datetime.now().isoformat()))

            conn.commit()

        except Exception as e:
            print(f"   DB error: {e}")

        finally:
            conn.close()

    # =========================================================================
    # SOURCE 1: GDELT Project (Free, extensive historical coverage)
    # =========================================================================
    def fetch_gdelt(self, symbol: str, date: str) -> List[Dict]:
        """
        Fetch news from GDELT Project.
        GDELT provides free access to global news mentions.

        Note: GDELT uses company names, not tickers, so we need mapping.
        """
        # Map tickers to company names for GDELT search
        ticker_to_name = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'NVDA': 'Nvidia',
            'META': 'Meta Facebook',
            'TSLA': 'Tesla',
            'JPM': 'JPMorgan',
            'BAC': 'Bank of America',
            'GS': 'Goldman Sachs',
        }

        company = ticker_to_name.get(symbol, symbol)

        try:
            # GDELT DOC API (free, no key required)
            # Format date for GDELT
            gdelt_date = date.replace('-', '')

            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': f'{company} stock market',
                'mode': 'artlist',
                'maxrecords': 50,
                'format': 'json',
                'startdatetime': f'{gdelt_date}000000',
                'enddatetime': f'{gdelt_date}235959',
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                return [{
                    'title': a.get('title', ''),
                    'url': a.get('url', ''),
                    'source': a.get('domain', 'gdelt'),
                    'date': date
                } for a in articles if a.get('title')]

        except Exception as e:
            pass  # GDELT can be unreliable

        return []

    # =========================================================================
    # SOURCE 2: NewsAPI.org (30 days free, good quality)
    # =========================================================================
    def fetch_newsapi(self, symbol: str, date: str) -> List[Dict]:
        """
        Fetch from NewsAPI.org.
        Free tier: 100 requests/day, 30 days history.
        """
        if not self.api_keys.get('newsapi'):
            return []

        ticker_to_name = {
            'SPY': 'S&P 500 OR SPY ETF',
            'QQQ': 'Nasdaq OR QQQ ETF',
            'AAPL': 'Apple stock',
            'MSFT': 'Microsoft stock',
            'GOOGL': 'Google Alphabet stock',
            'AMZN': 'Amazon stock',
            'NVDA': 'Nvidia stock',
            'META': 'Meta stock OR Facebook stock',
            'TSLA': 'Tesla stock',
        }

        query = ticker_to_name.get(symbol, f'{symbol} stock')

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': date,
                'to': date,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 50,
                'apiKey': self.api_keys['newsapi']
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                return [{
                    'title': a.get('title', ''),
                    'url': a.get('url', ''),
                    'source': a.get('source', {}).get('name', 'newsapi'),
                    'date': date
                } for a in articles if a.get('title')]

        except Exception as e:
            pass

        return []

    # =========================================================================
    # SOURCE 3: Wayback Machine (Historical snapshots)
    # =========================================================================
    def fetch_wayback_google_news(self, symbol: str, date: str) -> List[Dict]:
        """
        Fetch historical Google News via Wayback Machine.
        This is slower but provides deep historical coverage.
        """
        try:
            # Format: YYYYMMDD
            wayback_date = date.replace('-', '')

            # Try to get Google News search snapshot
            google_news_url = f"https://news.google.com/search?q={symbol}+stock"

            # Wayback CDX API to find snapshots
            cdx_url = "http://web.archive.org/cdx/search/cdx"
            params = {
                'url': google_news_url,
                'output': 'json',
                'from': wayback_date,
                'to': wayback_date,
                'limit': 1
            }

            response = requests.get(cdx_url, params=params, timeout=30)

            if response.status_code == 200 and response.text.strip():
                # Parse response (returns list of lists)
                data = response.json()
                if len(data) > 1:  # First row is headers
                    timestamp = data[1][1]  # Timestamp column
                    archive_url = f"http://web.archive.org/web/{timestamp}/{google_news_url}"

                    # Note: Actually parsing the archived page would require
                    # HTML parsing which is complex. For now, we just log
                    # that a snapshot exists.
                    return [{
                        'title': f'Wayback snapshot available for {symbol} on {date}',
                        'url': archive_url,
                        'source': 'wayback',
                        'date': date
                    }]

        except Exception as e:
            pass

        return []

    # =========================================================================
    # SOURCE 4: Polygon.io News (if API key available)
    # =========================================================================
    def fetch_polygon_news(self, symbol: str, date: str) -> List[Dict]:
        """Fetch from Polygon.io news endpoint."""
        if not self.api_keys.get('polygon'):
            return []

        try:
            url = f"https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': symbol,
                'published_utc.gte': f'{date}T00:00:00Z',
                'published_utc.lte': f'{date}T23:59:59Z',
                'limit': 50,
                'apiKey': self.api_keys['polygon']
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])

                return [{
                    'title': a.get('title', ''),
                    'url': a.get('article_url', ''),
                    'source': a.get('publisher', {}).get('name', 'polygon'),
                    'date': date
                } for a in articles if a.get('title')]

        except Exception as e:
            pass

        return []

    # =========================================================================
    # SOURCE 5: Yahoo Finance RSS (Recent news only)
    # =========================================================================
    def fetch_yahoo_rss(self, symbol: str) -> List[Dict]:
        """Fetch recent news from Yahoo Finance RSS."""
        import xml.etree.ElementTree as ET

        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                articles = []

                for item in root.findall('.//item')[:30]:
                    title = item.find('title')
                    link = item.find('link')
                    pub_date = item.find('pubDate')

                    if title is not None and title.text:
                        # Parse date from pubDate
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        if pub_date is not None and pub_date.text:
                            try:
                                from email.utils import parsedate_to_datetime
                                dt = parsedate_to_datetime(pub_date.text)
                                date_str = dt.strftime('%Y-%m-%d')
                            except:
                                pass

                        articles.append({
                            'title': title.text,
                            'url': link.text if link is not None else '',
                            'source': 'yahoo',
                            'date': date_str
                        })

                return articles

        except Exception as e:
            pass

        return []

    # =========================================================================
    # SINGLE SYMBOL BACKFILL (for parallel execution)
    # =========================================================================
    def _backfill_symbol(self, symbol: str, days: int, sources: List[str]) -> Dict:
        """Backfill a single symbol - used for parallel execution."""
        result = {'symbol': symbol, 'days': 0, 'articles': 0}

        for day_offset in range(days):
            date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            all_headlines = []

            for source in sources:
                time.sleep(0.02)  # Minimal rate limiting - GDELT handles high volume

                try:
                    if source == 'gdelt':
                        headlines = self.fetch_gdelt(symbol, date)
                    elif source == 'newsapi':
                        headlines = self.fetch_newsapi(symbol, date)
                    elif source == 'polygon':
                        headlines = self.fetch_polygon_news(symbol, date)
                    elif source == 'yahoo' and day_offset < 7:
                        headlines = self.fetch_yahoo_rss(symbol)
                        headlines = [h for h in headlines if h['date'] == date]
                    else:
                        headlines = []

                    if headlines:
                        all_headlines.extend(headlines)
                except:
                    pass

            if all_headlines:
                seen = set()
                unique_headlines = []
                for h in all_headlines:
                    title_key = h.get('title', '')[:50].lower()
                    if title_key not in seen:
                        seen.add(title_key)
                        unique_headlines.append(h)

                self.save_daily_sentiment(symbol, date, unique_headlines, 'backfill')
                result['days'] += 1
                result['articles'] += len(unique_headlines)

            # Progress update every 100 days
            if day_offset > 0 and day_offset % 100 == 0:
                print(f"   [{symbol}] {day_offset}/{days} days processed...")

        print(f"   [{symbol}] Complete: {result['days']} days, {result['articles']} articles")
        return result

    # =========================================================================
    # PARALLEL BACKFILL FUNCTION
    # =========================================================================
    def backfill_parallel(self, symbols: List[str], days: int = 30, max_workers: int = 5):
        """
        Backfill historical news for given symbols in PARALLEL.

        Args:
            symbols: List of ticker symbols
            days: Number of days to backfill
            max_workers: Number of parallel workers (default: 5)
        """
        print("=" * 70)
        print("HISTORICAL NEWS BACKFILL (PARALLEL)")
        print("=" * 70)
        print(f"Symbols: {len(symbols)}")
        print(f"Days: {days}")
        print(f"Workers: {max_workers}")
        print(f"Date Range: {(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 70)

        # Check available sources
        sources = ['gdelt']
        if self.api_keys.get('polygon'):
            sources.append('polygon')
            print("Polygon News: Available")
        print(f"Sources: {', '.join(sources)}")
        print("")

        start_time = time.time()
        results = {'total_days': 0, 'successful_days': 0, 'total_articles': 0, 'by_symbol': {}}

        # Run in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._backfill_symbol, symbol, days, sources): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results['by_symbol'][symbol] = result
                    results['successful_days'] += result['days']
                    results['total_articles'] += result['articles']
                    results['total_days'] += days
                except Exception as e:
                    print(f"   [{symbol}] Error: {e}")
                    results['by_symbol'][symbol] = {'days': 0, 'articles': 0}

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        print(f"Total Time: {elapsed/60:.1f} minutes")
        print(f"Total Days Processed: {results['total_days']}")
        print(f"Days with Data: {results['successful_days']}")
        print(f"Total Articles: {results['total_articles']}")
        print(f"Success Rate: {results['successful_days']/max(results['total_days'],1)*100:.1f}%")
        print("")
        print("By Symbol:")
        for symbol, data in sorted(results['by_symbol'].items()):
            print(f"  {symbol}: {data['days']} days, {data['articles']} articles")

        return results

    # =========================================================================
    # MAIN BACKFILL FUNCTION (Sequential - kept for compatibility)
    # =========================================================================
    def backfill(self, symbols: List[str], days: int = 30, verbose: bool = True):
        """
        Backfill historical news for given symbols (sequential).

        Args:
            symbols: List of ticker symbols
            days: Number of days to backfill
            verbose: Print progress
        """
        results = {
            'total_days': 0,
            'successful_days': 0,
            'total_articles': 0,
            'by_symbol': {}
        }

        print("=" * 70)
        print("HISTORICAL NEWS BACKFILL")
        print("=" * 70)
        print(f"Symbols: {len(symbols)}")
        print(f"Days: {days}")
        print(f"Date Range: {(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 70)

        # Check available sources
        sources = ['gdelt', 'yahoo']
        if self.api_keys.get('newsapi'):
            sources.append('newsapi')
            print("NewsAPI: Available")
        if self.api_keys.get('polygon'):
            sources.append('polygon')
            print("Polygon News: Available")
        print(f"Sources: {', '.join(sources)}")
        print("")

        for symbol in symbols:
            print(f"\n[{symbol}] Backfilling {days} days...")
            results['by_symbol'][symbol] = {'days': 0, 'articles': 0}

            for day_offset in range(days):
                date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                results['total_days'] += 1

                all_headlines = []

                # Try each source
                for source in sources:
                    time.sleep(0.3)  # Rate limiting

                    if source == 'gdelt':
                        headlines = self.fetch_gdelt(symbol, date)
                    elif source == 'newsapi':
                        headlines = self.fetch_newsapi(symbol, date)
                    elif source == 'polygon':
                        headlines = self.fetch_polygon_news(symbol, date)
                    elif source == 'yahoo' and day_offset < 7:
                        headlines = self.fetch_yahoo_rss(symbol)
                        headlines = [h for h in headlines if h['date'] == date]
                    else:
                        headlines = []

                    if headlines:
                        all_headlines.extend(headlines)

                # Save combined results
                if all_headlines:
                    # Deduplicate by title
                    seen = set()
                    unique_headlines = []
                    for h in all_headlines:
                        title_key = h.get('title', '')[:50].lower()
                        if title_key not in seen:
                            seen.add(title_key)
                            unique_headlines.append(h)

                    self.save_daily_sentiment(symbol, date, unique_headlines, 'backfill')
                    results['successful_days'] += 1
                    results['total_articles'] += len(unique_headlines)
                    results['by_symbol'][symbol]['days'] += 1
                    results['by_symbol'][symbol]['articles'] += len(unique_headlines)

                    if verbose and day_offset % 7 == 0:
                        print(f"   {date}: {len(unique_headlines)} articles")

        # Summary
        print("\n" + "=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        print(f"Total Days Processed: {results['total_days']}")
        print(f"Days with Data: {results['successful_days']}")
        print(f"Total Articles: {results['total_articles']}")
        print(f"Success Rate: {results['successful_days']/max(results['total_days'],1)*100:.1f}%")
        print("")
        print("By Symbol:")
        for symbol, data in results['by_symbol'].items():
            print(f"  {symbol}: {data['days']} days, {data['articles']} articles")

        return results


def main():
    parser = argparse.ArgumentParser(description='Backfill historical news data')
    parser.add_argument('symbols', nargs='*', default=None, help='Symbols to backfill')
    parser.add_argument('--days', type=int, default=30, help='Days of history (default: 30)')
    parser.add_argument('--all', action='store_true', help='Backfill all default tickers')
    parser.add_argument('--parallel', '-p', action='store_true', help='Run in parallel mode (5x faster)')
    parser.add_argument('--workers', '-w', type=int, default=5, help='Number of parallel workers (default: 5)')

    args = parser.parse_args()

    if args.all:
        symbols = DEFAULT_TICKERS
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = ['SPY', 'QQQ']  # Default

    backfiller = HistoricalNewsBackfill()

    if args.parallel:
        backfiller.backfill_parallel(symbols, args.days, max_workers=args.workers)
    else:
        backfiller.backfill(symbols, args.days)


if __name__ == "__main__":
    main()
