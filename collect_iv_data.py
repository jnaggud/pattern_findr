"""
Daily IV Data Collection Script

Run this script daily (or add to cron) to build historical IV data.
The cached IV data will be used by the range prediction model.

Usage:
    python collect_iv_data.py                    # Collect for default tickers
    python collect_iv_data.py SPY QQQ AAPL      # Collect for specific tickers
    python collect_iv_data.py --all              # Collect for all popular tickers

Scheduling (cron example - run at 4:30 PM EST after market close):
    30 16 * * 1-5 cd /path/to/Pattern_FindR && python collect_iv_data.py

The script will:
1. Fetch current IV data from Polygon API
2. Save to SQLite database for historical analysis
3. Calculate IV rank and percentile based on history
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict

# Default tickers to collect IV data for
DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']

# Extended list for --all flag
ALL_TICKERS = [
    # Major ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLY',
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM',
    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
    # Other large caps
    'JNJ', 'PG', 'UNH', 'V', 'MA', 'HD', 'DIS', 'NFLX',
    # Volatility
    'VXX', 'UVXY', 'SVXY'
]


def load_polygon_api_key() -> str:
    """Load Polygon API key from user_settings.json."""
    try:
        settings_path = os.path.join(os.path.dirname(__file__), 'user_settings.json')
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return settings.get('polygon_api_key', '')
    except Exception as e:
        print(f"Error loading API key: {e}")
    return ''


def collect_iv_for_ticker(ticker: str, polygon_manager, db) -> Dict:
    """
    Collect IV data for a single ticker.

    Returns:
        Dict with collection result or None on failure
    """
    try:
        # Get current price
        price_data = polygon_manager.get_price_data(ticker, limit=1)
        if price_data.empty:
            return {'ticker': ticker, 'status': 'error', 'message': 'No price data'}

        current_price = price_data['close'].iloc[-1]

        # Get IV data from options
        iv_data = polygon_manager.calculate_aggregate_iv(ticker, current_price)

        if iv_data and iv_data.get('iv_weighted', 0) > 0:
            # Calculate IV rank and percentile from historical data
            historical_iv = db.get_iv_data(ticker)

            if not historical_iv.empty and len(historical_iv) > 20:
                iv_values = historical_iv['iv_weighted'].dropna()
                current_iv = iv_data['iv_weighted']

                # IV Rank: Where is current IV relative to 1-year range
                iv_min = iv_values.min()
                iv_max = iv_values.max()
                if iv_max > iv_min:
                    iv_data['iv_rank'] = (current_iv - iv_min) / (iv_max - iv_min) * 100
                else:
                    iv_data['iv_rank'] = 50

                # IV Percentile: What % of historical values are below current
                iv_data['iv_percentile'] = (iv_values < current_iv).mean() * 100
            else:
                iv_data['iv_rank'] = 50
                iv_data['iv_percentile'] = 50

            # Save to database
            db.save_iv_data(ticker, iv_data)

            return {
                'ticker': ticker,
                'status': 'success',
                'iv_weighted': iv_data['iv_weighted'],
                'iv_rank': iv_data['iv_rank'],
                'iv_percentile': iv_data['iv_percentile'],
                'atm_iv': iv_data.get('atm_iv', 0)
            }
        else:
            return {'ticker': ticker, 'status': 'no_options', 'message': 'No options data available'}

    except Exception as e:
        return {'ticker': ticker, 'status': 'error', 'message': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Collect daily IV data')
    parser.add_argument('tickers', nargs='*', default=None, help='Tickers to collect (default: major ETFs and stocks)')
    parser.add_argument('--all', action='store_true', help='Collect for all popular tickers (~40)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Determine tickers to collect
    if args.all:
        tickers = ALL_TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_TICKERS

    print("=" * 60)
    print("DAILY IV DATA COLLECTION")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {len(tickers)}")
    print("=" * 60)

    # Load API key
    api_key = load_polygon_api_key()
    if not api_key:
        print("\nERROR: No Polygon API key found!")
        print("Add to user_settings.json: {\"polygon_api_key\": \"YOUR_KEY\"}")
        sys.exit(1)

    # Initialize managers
    try:
        from polygon_manager import PolygonManager
        from market_data_db import get_market_db

        polygon = PolygonManager(api_key)
        db = get_market_db()
    except Exception as e:
        print(f"\nERROR: Could not initialize: {e}")
        sys.exit(1)

    # Collect IV data
    results = {'success': 0, 'no_options': 0, 'error': 0}
    successful = []
    failed = []

    for i, ticker in enumerate(tickers):
        if args.verbose:
            print(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")

        result = collect_iv_for_ticker(ticker, polygon, db)

        if result['status'] == 'success':
            results['success'] += 1
            successful.append(result)
            if args.verbose:
                print(f"   IV={result['iv_weighted']:.2f}% | Rank={result['iv_rank']:.0f}% | Percentile={result['iv_percentile']:.0f}%")
        elif result['status'] == 'no_options':
            results['no_options'] += 1
            if args.verbose:
                print(f"   No options data")
        else:
            results['error'] += 1
            failed.append(result)
            if args.verbose:
                print(f"   Error: {result.get('message', 'Unknown')}")

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Success: {results['success']}/{len(tickers)}")
    print(f"No Options: {results['no_options']}")
    print(f"Errors: {results['error']}")

    if successful:
        print("\nSuccessful collections:")
        # Sort by IV
        successful.sort(key=lambda x: x['iv_weighted'], reverse=True)
        print(f"{'Ticker':<8} {'IV %':>8} {'Rank':>8} {'%ile':>8}")
        print("-" * 35)
        for r in successful[:15]:  # Top 15 by IV
            print(f"{r['ticker']:<8} {r['iv_weighted']:>8.2f} {r['iv_rank']:>8.0f} {r['iv_percentile']:>8.0f}")

    if failed and args.verbose:
        print("\nFailed collections:")
        for r in failed:
            print(f"  {r['ticker']}: {r.get('message', 'Unknown error')}")

    # Show DB stats
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    for ticker in ['SPY', 'QQQ', 'AAPL'][:min(3, len(tickers))]:
        stats = db.get_iv_stats(ticker)
        if stats['row_count'] > 0:
            print(f"{ticker}: {stats['row_count']} records from {stats['earliest_date']} to {stats['latest_date']}")
            print(f"       IV range: {stats['min_iv']:.1f}% - {stats['max_iv']:.1f}% (avg: {stats['avg_iv']:.1f}%)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    # Return success if at least 50% collected
    success_rate = results['success'] / len(tickers) if tickers else 0
    sys.exit(0 if success_rate >= 0.5 else 1)


if __name__ == "__main__":
    main()
