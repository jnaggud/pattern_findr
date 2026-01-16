"""
Leverage & Sentiment Indicator Explorer

Explores whether liquidation/leverage data (crypto) and sentiment data (stocks)
can improve predictive power for our models.

BTC Indicators (via Coinglass/APIs):
- Funding Rate: When extreme, shows over-leveraged positioning
- Open Interest: Total leveraged positions, changes signal liquidation risk
- Long/Short Ratio: Sentiment extremes often precede reversals

SPY Indicators:
- Put/Call Ratio: Options sentiment (fear vs greed)
- VIX: Volatility/fear gauge
- VIX Term Structure: Contango vs backwardation

Usage:
    python leverage_sentiment_explorer.py [--ticker BTC-USD|SPY] [--days 365]
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA FETCHING - CRYPTO (BTC)
# =============================================================================

def fetch_coinglass_funding_rates(symbol: str = "BTC", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical funding rates from Coinglass.
    Funding rate shows who's paying whom - extreme values indicate over-leverage.
    """
    print(f"  Fetching funding rates for {symbol}...")

    # Coinglass public API endpoint
    url = "https://open-api.coinglass.com/public/v2/funding_usd_history"
    params = {
        "symbol": symbol,
        "time_type": "all"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data'):
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['createTime'], unit='ms')
                df.set_index('date', inplace=True)
                df = df.sort_index()
                # Get average funding rate across exchanges
                if 'fundingRate' in df.columns:
                    df['funding_rate'] = df['fundingRate'].astype(float)
                return df[['funding_rate']].last(f'{days}D')
    except Exception as e:
        print(f"    Coinglass API error: {e}")

    # Fallback: Try alternative source or generate synthetic based on price action
    return pd.DataFrame()


def fetch_coinglass_open_interest(symbol: str = "BTC", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical open interest data.
    Rising OI + flat price = liquidation cascade brewing.
    """
    print(f"  Fetching open interest for {symbol}...")

    url = "https://open-api.coinglass.com/public/v2/open_interest_history"
    params = {
        "symbol": symbol,
        "time_type": "all"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data'):
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['createTime'], unit='ms')
                df.set_index('date', inplace=True)
                df = df.sort_index()
                if 'openInterest' in df.columns:
                    df['open_interest'] = df['openInterest'].astype(float)
                return df[['open_interest']].last(f'{days}D')
    except Exception as e:
        print(f"    Coinglass OI API error: {e}")

    return pd.DataFrame()


def fetch_coinglass_long_short_ratio(symbol: str = "BTC", days: int = 365) -> pd.DataFrame:
    """
    Fetch long/short ratio - extreme readings often precede reversals.
    > 2.0 = too many longs, < 0.5 = too many shorts
    """
    print(f"  Fetching long/short ratio for {symbol}...")

    url = "https://open-api.coinglass.com/public/v2/long_short_history"
    params = {
        "symbol": symbol,
        "time_type": "all"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data'):
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['createTime'], unit='ms')
                df.set_index('date', inplace=True)
                df = df.sort_index()
                if 'longShortRatio' in df.columns:
                    df['long_short_ratio'] = df['longShortRatio'].astype(float)
                return df[['long_short_ratio']].last(f'{days}D')
    except Exception as e:
        print(f"    Coinglass L/S API error: {e}")

    return pd.DataFrame()


def fetch_alternative_crypto_data(symbol: str = "BTC", days: int = 365) -> pd.DataFrame:
    """
    Alternative: Fetch from CryptoQuant or derive from price action.
    Uses Fear & Greed Index as a proxy for sentiment.
    """
    print(f"  Fetching Fear & Greed Index as sentiment proxy...")

    url = "https://api.alternative.me/fng/?limit=0&format=json"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                records = []
                for item in data['data']:
                    records.append({
                        'date': pd.to_datetime(item['timestamp'], unit='s'),
                        'fear_greed_index': int(item['value'])
                    })
                df = pd.DataFrame(records)
                df.set_index('date', inplace=True)
                df = df.sort_index()
                return df.last(f'{days}D')
    except Exception as e:
        print(f"    Fear & Greed API error: {e}")

    return pd.DataFrame()


def derive_leverage_indicators_from_price(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Derive synthetic leverage indicators from price action.
    When we can't get actual data, we estimate based on:
    - Volatility clustering (high vol often follows liquidations)
    - Volume spikes (liquidations cause volume spikes)
    - Price velocity (rapid moves trigger liquidations)
    """
    df = df_price.copy()

    # Volatility (proxy for liquidation aftermath)
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)

    # Volume spike ratio (liquidations cause volume spikes)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / df['volume_sma']

    # Price velocity (rapid moves = liquidation cascades)
    df['price_velocity'] = df['close'].pct_change(5)  # 5-day momentum

    # Estimated "leverage stress" - combines volatility and velocity
    df['leverage_stress'] = (
        df['volatility_20d'].rank(pct=True) * 0.5 +
        df['price_velocity'].abs().rank(pct=True) * 0.5
    )

    return df


# =============================================================================
# DATA FETCHING - STOCKS (SPY)
# =============================================================================

def fetch_vix_data(days: int = 365) -> pd.DataFrame:
    """
    Fetch VIX (fear gauge) data.
    High VIX = fear, often marks bottoms.
    Low VIX = complacency, often precedes corrections.
    """
    print(f"  Fetching VIX data...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)

        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if not vix.empty:
            # Handle multi-level columns from yfinance
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)

            df = pd.DataFrame()
            df['vix'] = vix['Close']
            df['vix_sma20'] = df['vix'].rolling(20).mean()
            df['vix_percentile'] = df['vix'].rolling(252).rank(pct=True)
            return df.dropna()
    except Exception as e:
        print(f"    VIX fetch error: {e}")

    return pd.DataFrame()


def fetch_vix_term_structure(days: int = 365) -> pd.DataFrame:
    """
    Fetch VIX term structure (VIX vs VIX3M).
    Backwardation (VIX > VIX3M) = fear/stress
    Contango (VIX < VIX3M) = complacency
    """
    print(f"  Fetching VIX term structure...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)

        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        vix3m = yf.download("^VIX3M", start=start_date, end=end_date, progress=False)

        if not vix.empty and not vix3m.empty:
            # Handle multi-level columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if isinstance(vix3m.columns, pd.MultiIndex):
                vix3m.columns = vix3m.columns.get_level_values(0)

            df = pd.DataFrame()
            df['vix'] = vix['Close']
            df['vix3m'] = vix3m['Close']
            df['vix_term_structure'] = df['vix'] / df['vix3m']  # < 1 = contango, > 1 = backwardation
            df['term_structure_zscore'] = (
                (df['vix_term_structure'] - df['vix_term_structure'].rolling(60).mean()) /
                df['vix_term_structure'].rolling(60).std()
            )
            return df.dropna()
    except Exception as e:
        print(f"    VIX term structure error: {e}")

    return pd.DataFrame()


def fetch_put_call_ratio(days: int = 365) -> pd.DataFrame:
    """
    Fetch CBOE Put/Call ratio.
    > 1.2 = excessive fear (contrarian bullish)
    < 0.7 = excessive greed (contrarian bearish)
    """
    print(f"  Fetching Put/Call ratio...")

    # CBOE doesn't have a free API, but we can try to get equity P/C from other sources
    # For now, we'll derive it from SPY options if available, or use a proxy

    try:
        # Try CBOE total P/C ratio (if available via alternative sources)
        # This is a placeholder - in production you'd use a proper data source

        # Alternative: Calculate from SPY options chain snapshots
        # For now, return empty and we'll use VIX as primary sentiment gauge
        pass
    except Exception as e:
        print(f"    Put/Call ratio error: {e}")

    return pd.DataFrame()


def derive_spy_sentiment_from_vix(df_vix: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sentiment indicators from VIX data.
    """
    df = df_vix.copy()

    # VIX regime (high fear vs low fear)
    df['vix_regime'] = np.where(df['vix'] > df['vix'].rolling(50).mean() * 1.2, 'high_fear',
                               np.where(df['vix'] < df['vix'].rolling(50).mean() * 0.8, 'low_fear', 'neutral'))

    # VIX spike detection (sudden fear = potential bottom)
    df['vix_change'] = df['vix'].pct_change()
    df['vix_spike'] = df['vix_change'] > df['vix_change'].rolling(20).mean() + 2 * df['vix_change'].rolling(20).std()

    # Mean reversion signal (extreme VIX tends to revert)
    df['vix_mean_reversion'] = -1 * (df['vix'] - df['vix'].rolling(50).mean()) / df['vix'].rolling(50).std()

    return df


# =============================================================================
# COMPOSITE INDICATORS
# =============================================================================

def create_crypto_leverage_composite(
    df_price: pd.DataFrame,
    df_funding: pd.DataFrame = None,
    df_oi: pd.DataFrame = None,
    df_ls_ratio: pd.DataFrame = None,
    df_fear_greed: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create a composite "Crypto Leverage Sentiment" indicator.

    Combines:
    - Funding rate extremes
    - Open Interest changes
    - Long/Short ratio
    - Fear & Greed Index
    - Derived price-based leverage stress

    Output: -1 (extreme bearish/over-leveraged longs) to +1 (extreme bullish/over-leveraged shorts)
    """
    df = df_price.copy()

    # Start with price-derived indicators
    df = derive_leverage_indicators_from_price(df)

    components = []

    # Merge funding rate if available
    if df_funding is not None and not df_funding.empty:
        df = df.join(df_funding, how='left')
        if 'funding_rate' in df.columns:
            # Normalize funding rate: positive = longs paying (bearish signal)
            df['funding_signal'] = -1 * df['funding_rate'].rank(pct=True).apply(lambda x: (x - 0.5) * 2)
            components.append('funding_signal')

    # Merge open interest if available
    if df_oi is not None and not df_oi.empty:
        df = df.join(df_oi, how='left')
        if 'open_interest' in df.columns:
            # OI change: rapid increase with flat price = bearish (liquidation risk)
            df['oi_change'] = df['open_interest'].pct_change(5)
            df['oi_signal'] = -1 * df['oi_change'].rank(pct=True).apply(lambda x: (x - 0.5) * 2)
            components.append('oi_signal')

    # Merge long/short ratio if available
    if df_ls_ratio is not None and not df_ls_ratio.empty:
        df = df.join(df_ls_ratio, how='left')
        if 'long_short_ratio' in df.columns:
            # High L/S ratio = too many longs = bearish signal
            df['ls_signal'] = -1 * df['long_short_ratio'].rank(pct=True).apply(lambda x: (x - 0.5) * 2)
            components.append('ls_signal')

    # Merge fear & greed if available
    if df_fear_greed is not None and not df_fear_greed.empty:
        df = df.join(df_fear_greed, how='left')
        if 'fear_greed_index' in df.columns:
            # High F&G = greed = bearish signal, Low = fear = bullish signal
            df['fg_signal'] = -1 * (df['fear_greed_index'] - 50) / 50
            components.append('fg_signal')

    # Always include price-derived leverage stress
    if 'leverage_stress' in df.columns:
        # High stress after up-move = bearish, after down-move = bullish
        df['stress_direction'] = np.sign(df['price_velocity'])
        df['stress_signal'] = -1 * df['leverage_stress'] * df['stress_direction']
        components.append('stress_signal')

    # Create composite
    if components:
        df['leverage_sentiment'] = df[components].mean(axis=1)
        # Normalize to -1 to 1
        df['leverage_sentiment'] = df['leverage_sentiment'].clip(-1, 1)
    else:
        df['leverage_sentiment'] = 0

    return df


def create_spy_sentiment_composite(
    df_price: pd.DataFrame,
    df_vix: pd.DataFrame = None,
    df_term_structure: pd.DataFrame = None,
    df_put_call: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create a composite "SPY Sentiment" indicator.

    Combines:
    - VIX level and regime
    - VIX term structure
    - Put/Call ratio (if available)

    Output: -1 (extreme fear/oversold) to +1 (extreme greed/overbought)
    """
    df = df_price.copy()
    components = []

    # Merge VIX data
    if df_vix is not None and not df_vix.empty:
        df = df.join(df_vix[['vix', 'vix_percentile']], how='left')
        if 'vix_percentile' in df.columns:
            # High VIX percentile = fear = bullish signal (contrarian)
            df['vix_signal'] = -1 * (df['vix_percentile'] - 0.5) * 2
            components.append('vix_signal')

    # Merge term structure
    if df_term_structure is not None and not df_term_structure.empty:
        df = df.join(df_term_structure[['vix_term_structure', 'term_structure_zscore']], how='left')
        if 'term_structure_zscore' in df.columns:
            # Backwardation (high z-score) = fear = bullish signal (contrarian)
            df['term_signal'] = -1 * df['term_structure_zscore'].clip(-2, 2) / 2
            components.append('term_signal')

    # Merge put/call ratio if available
    if df_put_call is not None and not df_put_call.empty and 'put_call_ratio' in df_put_call.columns:
        df = df.join(df_put_call[['put_call_ratio']], how='left')
        # High P/C = fear = bullish signal (contrarian)
        df['pc_signal'] = -1 * (df['put_call_ratio'].rank(pct=True) - 0.5) * 2
        components.append('pc_signal')

    # Create composite
    if components:
        df['sentiment_composite'] = df[components].mean(axis=1)
        df['sentiment_composite'] = df['sentiment_composite'].clip(-1, 1)
    else:
        df['sentiment_composite'] = 0

    return df


# =============================================================================
# ANALYSIS & TESTING
# =============================================================================

def analyze_predictive_power(df: pd.DataFrame, indicator_col: str, forward_days: list = [1, 5, 10, 20]) -> dict:
    """
    Analyze how well an indicator predicts future returns.
    """
    results = {}

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    for days in forward_days:
        df[f'fwd_return_{days}d'] = df['close'].pct_change(days).shift(-days)

        # Correlation
        corr = df[indicator_col].corr(df[f'fwd_return_{days}d'])

        # Quintile analysis (handle cases with few unique values)
        try:
            df['quintile'] = pd.qcut(df[indicator_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except ValueError:
            # Fall back to rank-based quintiles
            df['quintile'] = pd.cut(df[indicator_col].rank(method='first'), 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        quintile_returns = df.groupby('quintile')[f'fwd_return_{days}d'].mean()

        # Extreme signal performance
        extreme_bullish = df[df[indicator_col] < df[indicator_col].quantile(0.1)][f'fwd_return_{days}d'].mean()
        extreme_bearish = df[df[indicator_col] > df[indicator_col].quantile(0.9)][f'fwd_return_{days}d'].mean()

        results[f'{days}d'] = {
            'correlation': corr,
            'quintile_spread': quintile_returns.get('Q5', 0) - quintile_returns.get('Q1', 0) if len(quintile_returns) >= 2 else 0,
            'extreme_bullish_return': extreme_bullish,
            'extreme_bearish_return': extreme_bearish,
            'signal_value': extreme_bullish - extreme_bearish  # Positive = indicator works
        }

    return results


def plot_indicator_analysis(df: pd.DataFrame, indicator_col: str, ticker: str, title: str):
    """
    Create visualization of indicator vs price and returns.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{title} - {ticker}', fontsize=14, fontweight='bold')

    # Price
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], color='white', linewidth=1)
    ax1.set_ylabel('Price', color='white')
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)

    # Indicator
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in df[indicator_col]]
    ax2.bar(df.index, df[indicator_col], color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='yellow', linestyle=':', alpha=0.3)
    ax2.axhline(y=-0.5, color='yellow', linestyle=':', alpha=0.3)
    ax2.set_ylabel(indicator_col, color='white')
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)

    # Forward returns (5-day)
    df['fwd_5d'] = df['close'].pct_change(5).shift(-5) * 100
    ax3 = axes[2]
    ax3.plot(df.index, df['fwd_5d'], color='cyan', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax3.set_ylabel('5D Fwd Return %', color='white')
    ax3.set_facecolor('#1a1a2e')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2)

    # Scatter: Indicator vs Forward Return
    ax4 = axes[3]
    valid = df[[indicator_col, 'fwd_5d']].dropna()
    scatter_colors = ['green' if r > 0 else 'red' for r in valid['fwd_5d']]
    ax4.scatter(valid[indicator_col], valid['fwd_5d'], c=scatter_colors, alpha=0.3, s=10)
    ax4.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax4.set_xlabel(indicator_col, color='white')
    ax4.set_ylabel('5D Fwd Return %', color='white')
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.2)

    # Add correlation annotation
    corr = valid[indicator_col].corr(valid['fwd_5d'])
    ax4.annotate(f'Correlation: {corr:.3f}', xy=(0.02, 0.98), xycoords='axes fraction',
                 color='yellow', fontsize=10, verticalalignment='top')

    fig.patch.set_facecolor('#0d0d1a')
    plt.tight_layout()

    # Save
    filename = f'leverage_sentiment_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, facecolor='#0d0d1a', dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {filename}")
    plt.close()

    return filename


# =============================================================================
# MAIN EXPLORATION
# =============================================================================

def explore_btc_leverage_indicators(days: int = 365):
    """
    Explore BTC leverage/liquidation indicators.
    """
    print("\n" + "="*60)
    print("EXPLORING BTC LEVERAGE & SENTIMENT INDICATORS")
    print("="*60)

    # Fetch price data
    print("\nFetching BTC price data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)

    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    btc.columns = [c.lower() for c in btc.columns]

    if btc.empty:
        print("  ERROR: Could not fetch BTC price data")
        return None

    print(f"  Got {len(btc)} days of price data")

    # Fetch leverage indicators
    print("\nFetching leverage indicators...")
    df_funding = fetch_coinglass_funding_rates("BTC", days)
    df_oi = fetch_coinglass_open_interest("BTC", days)
    df_ls = fetch_coinglass_long_short_ratio("BTC", days)
    df_fg = fetch_alternative_crypto_data("BTC", days)

    # Create composite indicator
    print("\nCreating composite leverage sentiment indicator...")
    df = create_crypto_leverage_composite(btc, df_funding, df_oi, df_ls, df_fg)

    # Analyze predictive power
    print("\nAnalyzing predictive power...")
    results = analyze_predictive_power(df, 'leverage_sentiment')

    print("\n" + "-"*50)
    print("PREDICTIVE POWER ANALYSIS - BTC Leverage Sentiment")
    print("-"*50)
    for period, metrics in results.items():
        print(f"\n{period} Forward Returns:")
        print(f"  Correlation:           {metrics['correlation']:.4f}")
        print(f"  Quintile Spread:       {metrics['quintile_spread']*100:.2f}%")
        print(f"  Extreme Bullish Avg:   {metrics['extreme_bullish_return']*100:.2f}%")
        print(f"  Extreme Bearish Avg:   {metrics['extreme_bearish_return']*100:.2f}%")
        print(f"  Signal Value:          {metrics['signal_value']*100:.2f}%")

    # Create visualization
    print("\nGenerating visualization...")
    plot_indicator_analysis(df, 'leverage_sentiment', 'BTC-USD', 'Leverage Sentiment Indicator')

    # Show available components
    print("\n" + "-"*50)
    print("AVAILABLE INDICATOR COMPONENTS")
    print("-"*50)
    component_cols = ['funding_signal', 'oi_signal', 'ls_signal', 'fg_signal', 'stress_signal']
    for col in component_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} observations")

    return df


def explore_spy_sentiment_indicators(days: int = 365):
    """
    Explore SPY sentiment indicators.
    """
    print("\n" + "="*60)
    print("EXPLORING SPY SENTIMENT INDICATORS")
    print("="*60)

    # Fetch price data
    print("\nFetching SPY price data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)

    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]

    if spy.empty:
        print("  ERROR: Could not fetch SPY price data")
        return None

    print(f"  Got {len(spy)} days of price data")

    # Fetch sentiment indicators
    print("\nFetching sentiment indicators...")
    df_vix = fetch_vix_data(days)
    df_term = fetch_vix_term_structure(days)
    df_pc = fetch_put_call_ratio(days)

    # Create composite indicator
    print("\nCreating composite sentiment indicator...")
    df = create_spy_sentiment_composite(spy, df_vix, df_term, df_pc)

    # Also create VIX-based sentiment signals
    if df_vix is not None and not df_vix.empty:
        df_vix_sentiment = derive_spy_sentiment_from_vix(df_vix)
        for col in ['vix_mean_reversion', 'vix_spike']:
            if col in df_vix_sentiment.columns:
                df = df.join(df_vix_sentiment[[col]], how='left')

    # Analyze predictive power
    print("\nAnalyzing predictive power...")
    results = analyze_predictive_power(df, 'sentiment_composite')

    print("\n" + "-"*50)
    print("PREDICTIVE POWER ANALYSIS - SPY Sentiment Composite")
    print("-"*50)
    for period, metrics in results.items():
        print(f"\n{period} Forward Returns:")
        print(f"  Correlation:           {metrics['correlation']:.4f}")
        print(f"  Quintile Spread:       {metrics['quintile_spread']*100:.2f}%")
        print(f"  Extreme Bullish Avg:   {metrics['extreme_bullish_return']*100:.2f}%")
        print(f"  Extreme Bearish Avg:   {metrics['extreme_bearish_return']*100:.2f}%")
        print(f"  Signal Value:          {metrics['signal_value']*100:.2f}%")

    # Also analyze individual components
    if 'vix_mean_reversion' in df.columns:
        df_vix_subset = df.dropna(subset=['vix_mean_reversion'])
        if len(df_vix_subset) > 50:  # Need enough data for analysis
            print("\n" + "-"*50)
            print("PREDICTIVE POWER - VIX Mean Reversion Signal")
            print("-"*50)
            vix_results = analyze_predictive_power(df_vix_subset, 'vix_mean_reversion')
            for period, metrics in vix_results.items():
                print(f"  {period}: Corr={metrics['correlation']:.4f}, Signal={metrics['signal_value']*100:.2f}%")

    # Create visualization
    print("\nGenerating visualization...")
    plot_indicator_analysis(df, 'sentiment_composite', 'SPY', 'Sentiment Composite Indicator')

    # Show available components
    print("\n" + "-"*50)
    print("AVAILABLE INDICATOR COMPONENTS")
    print("-"*50)
    component_cols = ['vix_signal', 'term_signal', 'pc_signal', 'vix_mean_reversion']
    for col in component_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} observations")

    return df


def compare_indicators(df_btc: pd.DataFrame, df_spy: pd.DataFrame):
    """
    Compare the effectiveness of BTC vs SPY sentiment indicators.
    """
    print("\n" + "="*60)
    print("COMPARISON: BTC vs SPY SENTIMENT INDICATORS")
    print("="*60)

    btc_results = analyze_predictive_power(df_btc, 'leverage_sentiment') if df_btc is not None else {}
    spy_results = analyze_predictive_power(df_spy, 'sentiment_composite') if df_spy is not None else {}

    print("\n{:<20} {:>15} {:>15}".format("Metric", "BTC Leverage", "SPY Sentiment"))
    print("-"*50)

    for period in ['1d', '5d', '10d', '20d']:
        btc_val = btc_results.get(period, {}).get('signal_value', 0) * 100
        spy_val = spy_results.get(period, {}).get('signal_value', 0) * 100
        print(f"{period} Signal Value:    {btc_val:>14.2f}% {spy_val:>14.2f}%")

    print("\n" + "-"*50)
    print("INTERPRETATION:")
    print("-"*50)
    print("  - Positive Signal Value = Indicator has predictive power")
    print("  - Higher = Better predictive ability")
    print("  - Compare extreme bullish vs bearish returns")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore leverage and sentiment indicators")
    parser.add_argument("--ticker", type=str, default="both", help="BTC-USD, SPY, or both")
    parser.add_argument("--days", type=int, default=365, help="Days of history to analyze")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LEVERAGE & SENTIMENT INDICATOR EXPLORER")
    print("="*60)
    print(f"Analyzing {args.days} days of data")
    print(f"Ticker: {args.ticker}")

    df_btc = None
    df_spy = None

    if args.ticker.upper() in ["BTC-USD", "BTC", "BOTH"]:
        df_btc = explore_btc_leverage_indicators(args.days)

    if args.ticker.upper() in ["SPY", "BOTH"]:
        df_spy = explore_spy_sentiment_indicators(args.days)

    if args.ticker.upper() == "BOTH" and df_btc is not None and df_spy is not None:
        compare_indicators(df_btc, df_spy)

    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the generated charts")
    print("  2. If Signal Value is positive, consider integrating into models")
    print("  3. Test as additional feature in walk-forward validation")
    print("  4. Consider as filter (only trade when sentiment aligns)")
