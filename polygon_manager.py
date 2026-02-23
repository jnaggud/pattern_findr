"""
Polygon.io API Manager for Options and Price Data

Provides:
- OHLCV price data
- Options sentiment (Put/Call ratio)
- Options chain data with IV
- Max pain calculation
- High OI strike identification
- Greeks aggregation
- Term structure analysis
- Near-term expiration focus

Enhanced Features:
- Pagination support to get ALL options contracts
- SQLite-based persistent caching (survives restarts)
- Rate limiting protection
- Near-term expiration filtering for better range prediction
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json
import sqlite3
import os

# Global cache database path
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), 'options_cache.db')

class PolygonManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

        # In-memory cache (fast, for same session)
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = 300  # 5 minutes for in-memory

        # SQLite persistent cache (survives restarts, shared across instances)
        self._db_cache_duration = 3600  # 1 hour for options data (markets don't change that fast)
        self._init_db_cache()

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.15  # 150ms between requests (avoid rate limits)

    def _init_db_cache(self):
        """Initialize SQLite cache database."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS options_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    expiry REAL,
                    created_at TEXT
                )
            ''')
            # Clean up expired entries
            cursor.execute('DELETE FROM options_cache WHERE expiry < ?', (time.time(),))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"   Warning: Could not initialize options cache DB: {e}")

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_cached(self, cache_key: str):
        """Get value from in-memory cache if not expired."""
        if cache_key in self._cache:
            if time.time() < self._cache_expiry.get(cache_key, 0):
                return self._cache[cache_key]
        return None

    def _set_cached(self, cache_key: str, value, duration: int = None):
        """Set value in in-memory cache with expiry."""
        self._cache[cache_key] = value
        self._cache_expiry[cache_key] = time.time() + (duration or self._cache_duration)

    def _get_db_cached(self, cache_key: str):
        """Get value from SQLite persistent cache if not expired."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT data, expiry FROM options_cache WHERE cache_key = ?',
                (cache_key,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                data, expiry = row
                if time.time() < expiry:
                    return json.loads(data)
        except Exception as e:
            print(f"   Warning: DB cache read error: {e}")
        return None

    def _set_db_cached(self, cache_key: str, value, duration: int = None):
        """Set value in SQLite persistent cache with expiry."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()
            expiry = time.time() + (duration or self._db_cache_duration)
            data = json.dumps(value)
            cursor.execute('''
                INSERT OR REPLACE INTO options_cache (cache_key, data, expiry, created_at)
                VALUES (?, ?, ?, ?)
            ''', (cache_key, data, expiry, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"   Warning: DB cache write error: {e}")
        
    def get_price_data(self, ticker: str, multiplier: int = 1, timespan: str = 'day', 
                       limit: int = 500, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Polygon (Faster/Reliable than YF).
        timespan: 'minute', 'hour', 'day'
        start_date/end_date: YYYY-MM-DD
        """
        # Adjust ticker for Polygon (e.g., BTC-USD -> X:BTCUSD)
        poly_ticker = self._format_ticker(ticker)
        
        to_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        
        if start_date:
            from_date = start_date
        else:
            from_date = (datetime.now() - timedelta(days=limit*2)).strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{poly_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        url = f"{self.base_url}{endpoint}?adjusted=true&sort=asc&limit={limit}&apiKey={self.api_key}"
        
        try:
            resp = requests.get(url)
            data = resp.json()
            
            if 'results' in data:
                df = pd.DataFrame(data['results'])
                # Rename columns to match our ML model (v, vw, o, c, h, l, t, n)
                df = df.rename(columns={
                    'v': 'volume',
                    'o': 'open',
                    'c': 'close',
                    'h': 'high',
                    'l': 'low',
                    't': 'datetime'
                })
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df = df.set_index('datetime')
                return df[['open', 'high', 'low', 'close', 'volume']]
            else:
                print(f"⚠️ Polygon: No results for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Polygon API Error: {e}")
            return pd.DataFrame()

    def get_options_sentiment(self, ticker: str) -> dict:
        """
        Calculate Sentiment from Options Chain (Put/Call Ratio).
        Note: This works for Stocks/ETFs. For BTC, use 'BITO' as proxy?
        """
        # Options data requires a stock ticker (e.g. SPY, AAPL)
        # If ticker is crypto, return neutral or try proxy
        if "USD" in ticker or ":" in ticker:
            return {"status": "skipped", "reason": "Crypto not supported for Options"}
            
        try:
            # 1. Get Snapshot of all active options for underlying
            # Endpoint: /v3/snapshot/options/{underlyingAsset}
            # This returns the entire chain. It might be heavy.
            url = f"{self.base_url}/v3/snapshot/options/{ticker}?apiKey={self.api_key}&limit=250"
            
            resp = requests.get(url)
            data = resp.json()
            
            if 'results' not in data:
                return {"status": "no_data", "pcr_vol": 0, "pcr_oi": 0}
                
            total_put_vol = 0
            total_call_vol = 0
            total_put_oi = 0
            total_call_oi = 0
            
            for contract in data['results']:
                details = contract.get('details', {})
                type_ = details.get('contract_type') # 'put' or 'call'
                
                day = contract.get('day', {})
                vol = day.get('volume', 0)
                oi = contract.get('open_interest', 0)
                
                if type_ == 'put':
                    total_put_vol += vol
                    total_put_oi += oi
                elif type_ == 'call':
                    total_call_vol += vol
                    total_call_oi += oi
                    
            # Calculate Ratios
            pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            
            # Sentiment Interpretation
            # High PCR (>1.0) = Bearish Sentiment (Buying Puts)
            # Low PCR (<0.6) = Bullish Sentiment (Buying Calls)
            # Extreme High (>2.0) = Potential Reversal (Oversold)
            
            sentiment = "NEUTRAL"
            if pcr_vol > 1.2: sentiment = "BEARISH"
            elif pcr_vol < 0.7: sentiment = "BULLISH"
            
            return {
                "status": "ok",
                "pcr_volume": round(pcr_vol, 2),
                "pcr_oi": round(pcr_oi, 2),
                "total_volume": total_put_vol + total_call_vol,
                "sentiment": sentiment
            }
            
        except Exception as e:
            print(f"❌ Polygon Options Error: {e}")
            return {"status": "error"}

    def _format_ticker(self, ticker: str) -> str:
        # Convert YFinance style to Polygon
        # BTC-USD -> X:BTCUSD
        if "-USD" in ticker:
            return f"X:{ticker.replace('-', '')}"
        return ticker

    def get_options_chain(self, ticker: str, expiration: str = None,
                          limit: int = 250, near_term_only: bool = False,
                          max_pages: int = 10) -> List[Dict]:
        """
        Get full options chain with strikes, IV, OI, and Greeks.

        ENHANCED: Supports pagination to get ALL contracts, not just first 250.

        Args:
            ticker: Stock/ETF ticker (e.g., 'SPY', 'AAPL')
            expiration: Optional specific expiration date (YYYY-MM-DD)
            limit: Contracts per page (max 250 per API call)
            near_term_only: If True, only get contracts expiring within 45 days
            max_pages: Maximum number of pages to fetch (prevent runaway)

        Returns:
            List of contract dictionaries with details
        """
        if "USD" in ticker or ":" in ticker:
            return []  # Options not available for crypto

        # Use date-based cache key (same day = same data, reduces fetches significantly)
        today = datetime.now().strftime('%Y-%m-%d')
        cache_key = f"options_chain_{ticker}_{expiration}_{near_term_only}_{today}"

        # Level 1: Check in-memory cache (fastest)
        cached = self._get_cached(cache_key)
        if cached:
            print(f"   Using memory-cached options chain for {ticker} ({len(cached)} contracts)")
            return cached

        # Level 2: Check SQLite persistent cache (survives restarts)
        db_cached = self._get_db_cached(cache_key)
        if db_cached:
            print(f"   Using DB-cached options chain for {ticker} ({len(db_cached)} contracts)")
            # Also store in memory for faster subsequent access
            self._set_cached(cache_key, db_cached)
            return db_cached

        # Level 3: Fetch from API
        print(f"   Fetching fresh options chain for {ticker} from Polygon...")
        try:
            all_contracts = []
            next_url = None
            page = 0

            # Calculate near-term expiration cutoff
            if near_term_only:
                cutoff_date = (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')

            while page < max_pages:
                self._rate_limit()

                if next_url:
                    # Use pagination URL from previous response
                    url = next_url
                    params = {'apiKey': self.api_key}
                else:
                    # First request
                    url = f"{self.base_url}/v3/snapshot/options/{ticker}"
                    params = {
                        'apiKey': self.api_key,
                        'limit': limit
                    }
                    if expiration:
                        params['expiration_date'] = expiration
                    elif near_term_only:
                        # Filter to near-term expirations
                        params['expiration_date.lte'] = cutoff_date

                resp = requests.get(url, params=params)
                data = resp.json()

                if 'results' not in data or not data['results']:
                    break

                for contract in data['results']:
                    details = contract.get('details', {})
                    greeks = contract.get('greeks', {})
                    day = contract.get('day', {})
                    underlying = contract.get('underlying_asset', {})

                    all_contracts.append({
                        'ticker': details.get('ticker'),
                        'contract_type': details.get('contract_type'),  # 'call' or 'put'
                        'strike_price': details.get('strike_price'),
                        'expiration_date': details.get('expiration_date'),
                        'open_interest': contract.get('open_interest', 0),
                        'volume': day.get('volume', 0),
                        'last_price': day.get('close', 0),
                        'bid': contract.get('last_quote', {}).get('bid', 0),
                        'ask': contract.get('last_quote', {}).get('ask', 0),
                        'implied_volatility': contract.get('implied_volatility', 0),
                        'delta': greeks.get('delta', 0),
                        'gamma': greeks.get('gamma', 0),
                        'theta': greeks.get('theta', 0),
                        'vega': greeks.get('vega', 0),
                        # Additional fields for enhanced analysis
                        'underlying_price': underlying.get('price', 0),
                        'days_to_expiry': self._calculate_dte(details.get('expiration_date')),
                        'moneyness': self._calculate_moneyness(
                            details.get('strike_price', 0),
                            underlying.get('price', 0),
                            details.get('contract_type', 'call')
                        )
                    })

                # Check for next page
                next_url = data.get('next_url')
                if not next_url:
                    break

                page += 1
                print(f"   Fetched page {page}: {len(all_contracts)} contracts so far...")

            print(f"   Total options contracts fetched for {ticker}: {len(all_contracts)}")

            # Cache the results in both memory and SQLite
            self._set_cached(cache_key, all_contracts)
            self._set_db_cached(cache_key, all_contracts)

            return all_contracts

        except Exception as e:
            print(f"❌ Polygon Options Chain Error: {e}")
            return []

    def _calculate_dte(self, expiration_date: str) -> int:
        """Calculate days to expiration."""
        if not expiration_date:
            return 0
        try:
            exp = datetime.strptime(expiration_date, '%Y-%m-%d')
            return max(0, (exp - datetime.now()).days)
        except:
            return 0

    def _calculate_moneyness(self, strike: float, underlying: float, contract_type: str) -> str:
        """Calculate if option is ITM, ATM, or OTM."""
        if not strike or not underlying:
            return 'unknown'
        ratio = strike / underlying
        if 0.97 <= ratio <= 1.03:
            return 'ATM'
        if contract_type == 'call':
            return 'ITM' if strike < underlying else 'OTM'
        else:  # put
            return 'ITM' if strike > underlying else 'OTM'

    def calculate_aggregate_iv(self, ticker: str, current_price: float = None) -> Dict:
        """
        Calculate weighted average implied volatility from options chain.

        Weights based on:
        1. Proximity to ATM (higher weight for near-the-money)
        2. Open Interest (higher OI = more liquid/reliable)

        Args:
            ticker: Stock/ETF ticker
            current_price: Current underlying price (fetched if not provided)

        Returns:
            Dict with iv_weighted, iv_call, iv_put, iv_skew
        """
        contracts = self.get_options_chain(ticker)

        if not contracts:
            return {'available': False}

        # Get current price if not provided
        if current_price is None:
            price_data = self.get_price_data(ticker, limit=1)
            if price_data.empty:
                return {'available': False}
            current_price = price_data['close'].iloc[-1]

        call_ivs = []
        call_weights = []
        put_ivs = []
        put_weights = []

        for c in contracts:
            iv = c.get('implied_volatility', 0)
            strike = c.get('strike_price', 0)
            oi = c.get('open_interest', 0)
            contract_type = c.get('contract_type')

            if iv <= 0 or strike <= 0:
                continue

            # Calculate weight based on ATM proximity and OI
            distance_from_atm = abs(strike - current_price) / current_price
            atm_weight = max(0, 1 - distance_from_atm * 5)  # Peak at ATM
            oi_weight = np.log1p(oi) if oi > 0 else 0
            weight = atm_weight * oi_weight

            if weight <= 0:
                continue

            if contract_type == 'call':
                call_ivs.append(iv)
                call_weights.append(weight)
            elif contract_type == 'put':
                put_ivs.append(iv)
                put_weights.append(weight)

        # Calculate weighted averages
        iv_call = np.average(call_ivs, weights=call_weights) if call_ivs and call_weights else 0
        iv_put = np.average(put_ivs, weights=put_weights) if put_ivs and put_weights else 0
        iv_weighted = (iv_call + iv_put) / 2 if iv_call and iv_put else (iv_call or iv_put)

        # IV skew (put IV - call IV), positive means more fear
        iv_skew = iv_put - iv_call if iv_put and iv_call else 0

        return {
            'available': True,
            'iv_weighted': round(iv_weighted * 100, 2),  # Convert to percentage
            'iv_call': round(iv_call * 100, 2),
            'iv_put': round(iv_put * 100, 2),
            'iv_skew': round(iv_skew * 100, 2),
            'underlying_price': current_price,
            'num_contracts': len(contracts)
        }

    def get_max_pain(self, ticker: str, expiration: str = None) -> Dict:
        """
        Calculate max pain price from options OI.

        Max pain is the strike price where option buyers would lose
        the most money (and option sellers make the most).

        Args:
            ticker: Stock/ETF ticker
            expiration: Specific expiration date (optional)

        Returns:
            Dict with max_pain price and analysis
        """
        contracts = self.get_options_chain(ticker, expiration=expiration)

        if not contracts:
            return {'available': False}

        # Group by strike price
        strikes = {}
        for c in contracts:
            strike = c.get('strike_price', 0)
            if strike <= 0:
                continue

            if strike not in strikes:
                strikes[strike] = {'call_oi': 0, 'put_oi': 0}

            if c.get('contract_type') == 'call':
                strikes[strike]['call_oi'] += c.get('open_interest', 0)
            else:
                strikes[strike]['put_oi'] += c.get('open_interest', 0)

        if not strikes:
            return {'available': False}

        # Calculate pain at each strike
        strike_prices = sorted(strikes.keys())
        min_pain = float('inf')
        max_pain_strike = None

        for test_price in strike_prices:
            total_pain = 0

            for strike, data in strikes.items():
                # Call pain: if price > strike, call is ITM
                if test_price > strike:
                    call_pain = (test_price - strike) * data['call_oi'] * 100
                else:
                    call_pain = 0

                # Put pain: if price < strike, put is ITM
                if test_price < strike:
                    put_pain = (strike - test_price) * data['put_oi'] * 100
                else:
                    put_pain = 0

                total_pain += call_pain + put_pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        return {
            'available': True,
            'max_pain': max_pain_strike,
            'total_strikes': len(strike_prices),
            'strike_range': (min(strike_prices), max(strike_prices))
        }

    def get_high_oi_strikes(self, ticker: str, top_n: int = 5) -> Dict:
        """
        Get strikes with highest open interest (calls and puts separately).

        These often act as support/resistance levels.

        Args:
            ticker: Stock/ETF ticker
            top_n: Number of top strikes to return

        Returns:
            Dict with high OI call and put strikes
        """
        contracts = self.get_options_chain(ticker)

        if not contracts:
            return {'available': False}

        # Separate calls and puts
        calls = [c for c in contracts if c.get('contract_type') == 'call']
        puts = [c for c in contracts if c.get('contract_type') == 'put']

        # Sort by OI
        calls.sort(key=lambda x: x.get('open_interest', 0), reverse=True)
        puts.sort(key=lambda x: x.get('open_interest', 0), reverse=True)

        # Get top strikes
        top_call_strikes = []
        for c in calls[:top_n]:
            top_call_strikes.append({
                'strike': c.get('strike_price'),
                'oi': c.get('open_interest', 0),
                'volume': c.get('volume', 0),
                'iv': round(c.get('implied_volatility', 0) * 100, 2)
            })

        top_put_strikes = []
        for p in puts[:top_n]:
            top_put_strikes.append({
                'strike': p.get('strike_price'),
                'oi': p.get('open_interest', 0),
                'volume': p.get('volume', 0),
                'iv': round(p.get('implied_volatility', 0) * 100, 2)
            })

        return {
            'available': True,
            'high_oi_calls': top_call_strikes,
            'high_oi_puts': top_put_strikes,
            'highest_call_strike': top_call_strikes[0]['strike'] if top_call_strikes else None,
            'highest_put_strike': top_put_strikes[0]['strike'] if top_put_strikes else None
        }

    def get_options_summary(self, ticker: str) -> Dict:
        """
        Get comprehensive options summary for a ticker.

        Combines sentiment, IV, max pain, and high OI data.
        """
        if "USD" in ticker or ":" in ticker:
            return {'available': False, 'reason': 'Crypto not supported'}

        try:
            sentiment = self.get_options_sentiment(ticker)
            iv_data = self.calculate_aggregate_iv(ticker)
            max_pain = self.get_max_pain(ticker)
            high_oi = self.get_high_oi_strikes(ticker)

            return {
                'available': True,
                'ticker': ticker,
                'sentiment': sentiment,
                'implied_volatility': iv_data,
                'max_pain': max_pain,
                'high_oi_strikes': high_oi,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    # =========================================================================
    # ENHANCED DATA METHODS (Added for comprehensive options analysis)
    # =========================================================================

    def get_term_structure(self, ticker: str) -> Dict:
        """
        Get IV term structure (IV by expiration date).

        Important for understanding volatility expectations at different horizons.

        Returns:
            Dict with iv_by_expiration list and term_structure_slope
        """
        contracts = self.get_options_chain(ticker, near_term_only=True)

        if not contracts:
            return {'available': False}

        try:
            # Group by expiration and calculate weighted IV
            expirations = {}
            for c in contracts:
                exp = c.get('expiration_date')
                if not exp:
                    continue

                dte = c.get('days_to_expiry', 0)
                iv = c.get('implied_volatility', 0)
                oi = c.get('open_interest', 0)

                if iv <= 0:
                    continue

                if exp not in expirations:
                    expirations[exp] = {'ivs': [], 'weights': [], 'dte': dte}

                expirations[exp]['ivs'].append(iv)
                expirations[exp]['weights'].append(max(1, np.log1p(oi)))

            # Calculate weighted average IV for each expiration
            term_structure = []
            for exp, data in sorted(expirations.items()):
                if data['ivs'] and data['weights']:
                    weighted_iv = np.average(data['ivs'], weights=data['weights'])
                    term_structure.append({
                        'expiration': exp,
                        'days_to_expiry': data['dte'],
                        'iv': round(weighted_iv * 100, 2),
                        'num_contracts': len(data['ivs'])
                    })

            # Calculate slope (contango/backwardation)
            # Positive slope = higher IV for further expirations (normal)
            # Negative slope = higher IV for near-term (fear/uncertainty)
            slope = 0
            if len(term_structure) >= 2:
                near_iv = term_structure[0]['iv']
                far_iv = term_structure[-1]['iv']
                slope = (far_iv - near_iv) / near_iv * 100 if near_iv > 0 else 0

            return {
                'available': True,
                'term_structure': term_structure,
                'slope_pct': round(slope, 2),
                'structure_type': 'contango' if slope > 5 else ('backwardation' if slope < -5 else 'flat'),
                'near_term_iv': term_structure[0]['iv'] if term_structure else 0,
                'far_term_iv': term_structure[-1]['iv'] if term_structure else 0
            }

        except Exception as e:
            print(f"❌ Term Structure Error: {e}")
            return {'available': False, 'error': str(e)}

    def get_greeks_aggregation(self, ticker: str, current_price: float = None) -> Dict:
        """
        Calculate aggregate Greeks (total delta, gamma exposure).

        Useful for understanding market maker positioning and potential
        support/resistance from gamma hedging.

        Returns:
            Dict with net_delta, net_gamma, gamma_exposure, delta_by_strike
        """
        contracts = self.get_options_chain(ticker, near_term_only=True)

        if not contracts:
            return {'available': False}

        # Get current price if not provided
        if current_price is None:
            price_data = self.get_price_data(ticker, limit=1)
            if price_data.empty:
                return {'available': False}
            current_price = price_data['close'].iloc[-1]

        try:
            total_call_delta = 0
            total_put_delta = 0
            total_gamma = 0
            gamma_by_strike = {}

            for c in contracts:
                delta = c.get('delta', 0)
                gamma = c.get('gamma', 0)
                oi = c.get('open_interest', 0)
                strike = c.get('strike_price', 0)
                contract_type = c.get('contract_type')

                if oi == 0:
                    continue

                # Calculate notional exposure (OI * 100 shares * delta)
                notional_delta = oi * 100 * delta

                if contract_type == 'call':
                    total_call_delta += notional_delta
                else:
                    total_put_delta += notional_delta

                # Gamma exposure (dollars gamma - how much delta changes per $1 move)
                gamma_notional = oi * 100 * gamma * current_price
                total_gamma += gamma_notional

                # Track gamma by strike
                if strike > 0:
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    gamma_by_strike[strike] += gamma_notional

            # Find high gamma strikes (potential support/resistance)
            sorted_gamma = sorted(gamma_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            return {
                'available': True,
                'net_delta': round(total_call_delta + total_put_delta, 0),
                'call_delta': round(total_call_delta, 0),
                'put_delta': round(total_put_delta, 0),
                'net_gamma': round(total_gamma, 0),
                'gamma_interpretation': 'long gamma' if total_gamma > 0 else 'short gamma',
                'high_gamma_strikes': [{'strike': s, 'gamma': round(g, 0)} for s, g in sorted_gamma],
                'current_price': current_price
            }

        except Exception as e:
            print(f"❌ Greeks Aggregation Error: {e}")
            return {'available': False, 'error': str(e)}

    def get_unusual_activity(self, ticker: str, volume_threshold: float = 2.0) -> Dict:
        """
        Detect unusual options activity (high volume relative to OI).

        Volume > 2x OI often signals new positions being opened.

        Args:
            ticker: Stock ticker
            volume_threshold: Volume/OI ratio threshold for "unusual"

        Returns:
            Dict with unusual_contracts list
        """
        contracts = self.get_options_chain(ticker, near_term_only=True)

        if not contracts:
            return {'available': False}

        try:
            unusual = []
            total_call_volume = 0
            total_put_volume = 0

            for c in contracts:
                volume = c.get('volume', 0)
                oi = c.get('open_interest', 0)
                contract_type = c.get('contract_type')

                # Track total volume
                if contract_type == 'call':
                    total_call_volume += volume
                else:
                    total_put_volume += volume

                # Check for unusual activity
                if oi > 0 and volume > 0:
                    ratio = volume / oi
                    if ratio >= volume_threshold:
                        unusual.append({
                            'ticker': c.get('ticker'),
                            'strike': c.get('strike_price'),
                            'expiration': c.get('expiration_date'),
                            'type': contract_type,
                            'volume': volume,
                            'oi': oi,
                            'vol_oi_ratio': round(ratio, 2),
                            'iv': round(c.get('implied_volatility', 0) * 100, 2)
                        })

            # Sort by volume/OI ratio
            unusual.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)

            # Determine bias from unusual activity
            unusual_calls = [u for u in unusual if u['type'] == 'call']
            unusual_puts = [u for u in unusual if u['type'] == 'put']

            if len(unusual_calls) > len(unusual_puts) * 1.5:
                activity_bias = 'BULLISH'
            elif len(unusual_puts) > len(unusual_calls) * 1.5:
                activity_bias = 'BEARISH'
            else:
                activity_bias = 'NEUTRAL'

            return {
                'available': True,
                'unusual_contracts': unusual[:20],  # Top 20
                'total_unusual': len(unusual),
                'unusual_calls': len(unusual_calls),
                'unusual_puts': len(unusual_puts),
                'activity_bias': activity_bias,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'pcr_volume': round(total_put_volume / total_call_volume, 2) if total_call_volume > 0 else 1.0
            }

        except Exception as e:
            print(f"❌ Unusual Activity Error: {e}")
            return {'available': False, 'error': str(e)}

    def get_atm_iv(self, ticker: str, current_price: float = None) -> Dict:
        """
        Get ATM (at-the-money) implied volatility.

        ATM IV is the most relevant for predicting near-term range.

        Returns:
            Dict with atm_call_iv, atm_put_iv, atm_avg_iv
        """
        contracts = self.get_options_chain(ticker, near_term_only=True)

        if not contracts:
            return {'available': False}

        # Get current price if not provided
        if current_price is None:
            price_data = self.get_price_data(ticker, limit=1)
            if price_data.empty:
                return {'available': False}
            current_price = price_data['close'].iloc[-1]

        try:
            # Find ATM contracts (within 1% of current price)
            atm_threshold = 0.01  # 1%
            atm_calls = []
            atm_puts = []

            for c in contracts:
                strike = c.get('strike_price', 0)
                iv = c.get('implied_volatility', 0)
                oi = c.get('open_interest', 0)

                if not strike or iv <= 0:
                    continue

                # Check if ATM
                pct_from_atm = abs(strike - current_price) / current_price
                if pct_from_atm <= atm_threshold:
                    if c.get('contract_type') == 'call':
                        atm_calls.append({'iv': iv, 'oi': oi, 'strike': strike})
                    else:
                        atm_puts.append({'iv': iv, 'oi': oi, 'strike': strike})

            # Calculate OI-weighted average IV for ATM options
            atm_call_iv = 0
            atm_put_iv = 0

            if atm_calls:
                weights = [c['oi'] for c in atm_calls]
                if sum(weights) > 0:
                    atm_call_iv = np.average([c['iv'] for c in atm_calls], weights=weights)
                else:
                    atm_call_iv = np.mean([c['iv'] for c in atm_calls])

            if atm_puts:
                weights = [p['oi'] for p in atm_puts]
                if sum(weights) > 0:
                    atm_put_iv = np.average([p['iv'] for p in atm_puts], weights=weights)
                else:
                    atm_put_iv = np.mean([p['iv'] for p in atm_puts])

            atm_avg_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else (atm_call_iv or atm_put_iv)

            return {
                'available': True,
                'atm_call_iv': round(atm_call_iv * 100, 2),
                'atm_put_iv': round(atm_put_iv * 100, 2),
                'atm_avg_iv': round(atm_avg_iv * 100, 2),
                'atm_skew': round((atm_put_iv - atm_call_iv) * 100, 2),
                'num_atm_calls': len(atm_calls),
                'num_atm_puts': len(atm_puts),
                'current_price': current_price
            }

        except Exception as e:
            print(f"❌ ATM IV Error: {e}")
            return {'available': False, 'error': str(e)}

    def get_full_options_analysis(self, ticker: str) -> Dict:
        """
        Get comprehensive options analysis with all available data.

        Combines all methods for maximum data extraction.

        Returns:
            Dict with all options metrics
        """
        if "USD" in ticker or ":" in ticker:
            return {'available': False, 'reason': 'Crypto not supported'}

        print(f"   Fetching comprehensive options data for {ticker}...")

        try:
            # Get current price first
            price_data = self.get_price_data(ticker, limit=1)
            current_price = price_data['close'].iloc[-1] if not price_data.empty else None

            # Fetch all data (uses caching internally)
            sentiment = self.get_options_sentiment(ticker)
            iv_data = self.calculate_aggregate_iv(ticker, current_price)
            atm_iv = self.get_atm_iv(ticker, current_price)
            max_pain = self.get_max_pain(ticker)
            high_oi = self.get_high_oi_strikes(ticker)
            term_structure = self.get_term_structure(ticker)
            greeks = self.get_greeks_aggregation(ticker, current_price)
            unusual = self.get_unusual_activity(ticker)

            return {
                'available': True,
                'ticker': ticker,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),

                # Core sentiment
                'pcr_volume': sentiment.get('pcr_volume', 1.0),
                'pcr_oi': sentiment.get('pcr_oi', 1.0),
                'sentiment': sentiment.get('sentiment', 'NEUTRAL'),

                # Implied Volatility
                'iv_weighted': iv_data.get('iv_weighted', 0),
                'iv_call': iv_data.get('iv_call', 0),
                'iv_put': iv_data.get('iv_put', 0),
                'iv_skew': iv_data.get('iv_skew', 0),
                'atm_iv': atm_iv.get('atm_avg_iv', 0),
                'atm_skew': atm_iv.get('atm_skew', 0),

                # Term Structure
                'term_structure_slope': term_structure.get('slope_pct', 0),
                'term_structure_type': term_structure.get('structure_type', 'unknown'),
                'near_term_iv': term_structure.get('near_term_iv', 0),
                'far_term_iv': term_structure.get('far_term_iv', 0),

                # Key Levels
                'max_pain': max_pain.get('max_pain'),
                'highest_call_oi_strike': high_oi.get('highest_call_strike'),
                'highest_put_oi_strike': high_oi.get('highest_put_strike'),

                # Greeks
                'net_delta': greeks.get('net_delta', 0),
                'net_gamma': greeks.get('net_gamma', 0),
                'gamma_interpretation': greeks.get('gamma_interpretation', 'unknown'),

                # Unusual Activity
                'activity_bias': unusual.get('activity_bias', 'NEUTRAL'),
                'unusual_contracts_count': unusual.get('total_unusual', 0),

                # Raw data for custom analysis
                'raw': {
                    'sentiment': sentiment,
                    'iv_data': iv_data,
                    'atm_iv': atm_iv,
                    'max_pain': max_pain,
                    'high_oi': high_oi,
                    'term_structure': term_structure,
                    'greeks': greeks,
                    'unusual': unusual
                }
            }

        except Exception as e:
            print(f"❌ Full Options Analysis Error: {e}")
            return {'available': False, 'error': str(e)}

    def compute_influence_zones(self, ticker: str = 'SPY') -> dict:
        """
        Compute options influence zones for use as trading filters.

        Calls get_full_options_analysis() and normalizes key metrics
        into three zone values, each in [-1, +1].

        Returns:
            Dict with:
                available: bool
                max_pain_zone: float [-1, +1] — price vs max pain
                gamma_zone: float [-1, +1] — dealer gamma positioning
                wall_zone: float [-1, +1] — proximity to call/put walls
                combined_zone: float [-1, +1] — weighted average
                raw: dict — full analysis for storage
        """
        analysis = self.get_full_options_analysis(ticker)
        if not analysis.get('available'):
            return {'available': False}

        spy_price = analysis.get('current_price', 0)
        max_pain = analysis.get('max_pain')
        net_gamma = analysis.get('net_gamma', 0)
        call_wall = analysis.get('highest_call_oi_strike')
        put_wall = analysis.get('highest_put_oi_strike')

        zones = {}

        # Zone 1: Max Pain Proximity [-1, +1]
        # Positive = price above max pain (bearish gravity)
        # Negative = price below max pain (bullish gravity)
        if max_pain and spy_price:
            mp_dist_pct = (spy_price - max_pain) / spy_price * 100
            zones['max_pain_zone'] = float(np.clip(mp_dist_pct / 2.0, -1, 1))
        else:
            zones['max_pain_zone'] = 0.0

        # Zone 2: Gamma Zone [-1, +1]
        # Positive = long gamma (dealers stabilize, mean reversion favored)
        # Negative = short gamma (dealers amplify, trend continuation)
        if net_gamma != 0:
            zones['gamma_zone'] = float(np.clip(net_gamma / 5e9, -1, 1))
        else:
            zones['gamma_zone'] = 0.0

        # Zone 3: Wall Proximity [-1, +1]
        # Positive = near call wall (resistance overhead)
        # Negative = near put wall (support below)
        if call_wall and put_wall and spy_price:
            call_dist = (call_wall - spy_price) / spy_price * 100
            put_dist = (spy_price - put_wall) / spy_price * 100
            if call_dist < put_dist:
                zones['wall_zone'] = float(np.clip(1.0 - call_dist / 3.0, -1, 1))
            else:
                zones['wall_zone'] = float(np.clip(-(1.0 - put_dist / 3.0), -1, 1))
        else:
            zones['wall_zone'] = 0.0

        # Combined zone: weighted average
        zones['combined_zone'] = float(np.clip(
            (zones['max_pain_zone'] + zones['gamma_zone'] + zones['wall_zone']) / 3.0,
            -1, 1
        ))

        return {
            'available': True,
            'zones': zones,
            **zones,
            **analysis,
        }
