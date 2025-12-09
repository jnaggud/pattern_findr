import requests
import pandas as pd
from datetime import datetime, timedelta

class PolygonManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
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
