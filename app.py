import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import json
import os
from datetime import datetime
import random
import tensorflow as tf
from dl_pattern_detector import create_chart_image, generate_training_data, build_cnn_model, load_and_preprocess_image, use_synthetic_data_for_training
from backtester import Backtester
from optimization import run_optimization
from indicators import get_all_indicators

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Streamlit for wide layout
st.set_page_config(
    page_title="Pattern_FindR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Pattern_FindR - Professional Trading Strategy Discovery")

def calculate_buy_and_hold_baseline(data, starting_capital=100000):
    """
    Calculate buy-and-hold strategy performance for baseline comparison.
    """
    if len(data) == 0:
        return {'total_return_pct': 0, 'ending_capital': starting_capital}
    
    entry_price = data['close'].iloc[0]
    exit_price = data['close'].iloc[-1]
    shares = starting_capital / entry_price
    ending_capital = shares * exit_price
    total_return_pct = ((ending_capital - starting_capital) / starting_capital) * 100
    
    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'ending_capital': ending_capital,
        'total_return_pct': total_return_pct,
        'starting_capital': starting_capital
    }

def create_strategy_chart(data, trade_log, strategy_name, summary, baseline=None):
    """
    Create an interactive chart showing price and actual trade entries/exits.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'{strategy_name} - Price & Trades', 'Portfolio Value Evolution'],
        row_heights=[0.55, 0.45]  # Give more space to portfolio chart
    )
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Remove buy/sell signals to declutter the chart - show only actual trades
    
    # Add executed trades from trade_log
    if len(trade_log) > 0:
        # Entry points
        entry_dates = pd.to_datetime(trade_log['entry_date'])
        entry_prices = trade_log['entry_price']
        
        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-up', 
                    size=14, 
                    color='blue',  # Changed from green to blue to contrast with candlesticks
                    line=dict(width=3, color='white')
                ),
                name='📈 BUY (Entry)',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Exit points
        exit_dates = pd.to_datetime(trade_log['exit_date'])
        exit_prices = trade_log['exit_price']
        
        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-down', 
                    size=14, 
                    color='orange',  # Changed from red to orange to contrast with candlesticks
                    line=dict(width=3, color='white')
                ),
                name='📉 SELL (Exit)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Calculate portfolio value curve properly
    portfolio_dates = data.index
    portfolio_value = []
    
    # Try to use backtester's equity curve first
    if 'equity_curve' in summary and len(summary['equity_curve']) >= len(data):
        # Use the backtester's equity curve (most accurate)
        portfolio_value = summary['equity_curve'][:len(data)]
        print(f"Using backtester equity curve: {len(portfolio_value)} points")
    elif len(trade_log) > 0:
        # Recalculate if backtester curve isn't available
        print("Recalculating portfolio value from trade log")
        current_capital = summary['starting_capital']
        position = None
        position_size = 0
        
        # Sort trades for proper processing
        trades_sorted = trade_log.sort_values('entry_date')
        trade_idx = 0
        
        for i, date in enumerate(data.index):
            current_price = data.loc[date, 'close']
            
            # Process any trade entries on this date
            while (trade_idx < len(trades_sorted) and 
                   pd.to_datetime(trades_sorted.iloc[trade_idx]['entry_date']).date() == date.date()):
                
                trade = trades_sorted.iloc[trade_idx]
                if position is None:  # Enter new position
                    entry_price = trade['entry_price']
                    position_size = current_capital / entry_price
                    position = {
                        'entry_date': trade['entry_date'],
                        'size': position_size,
                        'entry_price': entry_price
                    }
                    current_capital = 0  # All capital invested
                
                trade_idx += 1
            
            # Process any trade exits on this date
            if position is not None:
                # Check if we exit position today
                exit_trades = trades_sorted[
                    (pd.to_datetime(trades_sorted['exit_date']).dt.date == date.date())
                ]
                
                for _, exit_trade in exit_trades.iterrows():
                    if pd.to_datetime(exit_trade['entry_date']) == position['entry_date']:
                        # Exit position
                        exit_price = exit_trade['exit_price']
                        current_capital = position_size * exit_price
                        position = None
                        position_size = 0
                        break
            
            # Calculate portfolio value for this date
            if position is not None:
                # We're in a position - mark to market
                portfolio_val = position_size * current_price
            else:
                # We're in cash
                portfolio_val = current_capital
                
            portfolio_value.append(portfolio_val)
    else:
        # No trades - flat line at starting capital
        portfolio_value = [summary['starting_capital']] * len(data)
        print(f"No trades found - using flat line at ${summary['starting_capital']:,.2f}")
    
    # Ensure we have the right number of points
    if len(portfolio_value) != len(data):
        print(f"Warning: Portfolio value length ({len(portfolio_value)}) != data length ({len(data)})")
        # Pad or trim to match
        if len(portfolio_value) < len(data):
            # Extend with last value
            last_val = portfolio_value[-1] if portfolio_value else summary['starting_capital']
            portfolio_value.extend([last_val] * (len(data) - len(portfolio_value)))
        else:
            # Trim to match
            portfolio_value = portfolio_value[:len(data)]
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_dates,
            y=portfolio_value,
            mode='lines',
            line=dict(color='purple', width=4),
            fill='tonexty' if baseline is not None else 'tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)',
            name='Strategy Portfolio',
            showlegend=True,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add buy-and-hold baseline line if provided
    if baseline is not None:
        # Calculate daily buy-and-hold value
        baseline_shares = baseline['starting_capital'] / data.iloc[0]['close']
        baseline_daily_value = [baseline_shares * price for price in data['close']]
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=baseline_daily_value,
                mode='lines',
                line=dict(color='gray', width=3, dash='dash'),
                name='Buy & Hold Baseline',
                showlegend=True,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add starting capital line
    fig.add_hline(
        y=summary['starting_capital'],
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Capital",
        annotation_position="left",
        row=2, col=1
    )
    
    # Update layout for better readability
    fig.update_layout(
        title=dict(
            text=f"{strategy_name} - Performance Visualization",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Portfolio Value ($)",
        height=900,  # Increased height
        width=None,  # Use full container width
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=60, t=120, b=60),
        hovermode='x unified'  # Better hover interaction
    )
    
    # Format the portfolio value y-axis with better scaling
    if portfolio_value:
        min_val = min(portfolio_value)
        max_val = max(portfolio_value)
        
        # Use more generous padding for better visibility
        value_range = max_val - min_val
        if value_range > 0:
            padding = value_range * 0.15  # 15% padding for better visibility
        else:
            padding = max_val * 0.05  # 5% of max value if flat line
        
        fig.update_yaxes(
            title_text="Portfolio Value ($)",
            tickformat='$,.0f',
            range=[max(0, min_val - padding), max_val + padding],  # Don't go below $0
            nticks=8,  # More tick marks for better readability
            row=2, col=1
        )
    else:
        fig.update_yaxes(
            title_text="Portfolio Value ($)",
            tickformat='$,.0f',
            row=2, col=1
        )
    
    # Remove x-axis labels from top subplot
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    
    # Debug: Print portfolio value range
    if portfolio_value:
        print(f"Portfolio value range: ${min(portfolio_value):,.2f} to ${max(portfolio_value):,.2f}")
        print(f"Starting capital: ${summary['starting_capital']:,.2f}")
        print(f"Ending capital: ${summary['ending_capital']:,.2f}")
    
    return fig

def save_strategy_to_storage(strategy_data):
    """Save a strategy to persistent storage"""
    storage_dir = "saved_strategies"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategy_{timestamp}_{strategy_data['rank']}.json"
    filepath = os.path.join(storage_dir, filename)
    
    # Prepare data for JSON serialization
    serializable_data = {
        'timestamp': timestamp,
        'rank': strategy_data['rank'],
        'name': strategy_data['name'],
        'performance': {
            'total_return_pct': strategy_data['summary']['total_return_pct'],
            'total_trades': strategy_data['summary']['total_trades'],
            'total_profit': strategy_data['summary']['total_profit'],
            'win_rate': strategy_data['summary']['win_rate'],
            'profit_factor': strategy_data['summary']['profit_factor'],
            'max_drawdown_pct': strategy_data['summary']['max_drawdown_pct'],
            'ending_capital': strategy_data['summary']['ending_capital'],
            'starting_capital': strategy_data['summary']['starting_capital']
        },
        'parameters': strategy_data['trial'].params,
        'active_indicators': strategy_data['active_indicators'],
        'trade_log': strategy_data['trade_log'].copy().to_dict('records') if len(strategy_data['trade_log']) > 0 else [],
        'baseline_return': strategy_data['baseline']['total_return_pct'] if strategy_data['baseline'] else None
    }
    
    # Save to JSON file with custom serialization
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2, default=str)
    
    return filepath

def load_saved_strategies(sort_by='timestamp', ascending=False, min_return=None, max_return=None, search_term=''):
    """Load all saved strategies from storage with filtering and sorting options"""
    storage_dir = "saved_strategies"
    if not os.path.exists(storage_dir):
        return [], []
    
    strategies = []
    errors = []
    
    for filename in os.listdir(storage_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(storage_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    strategy_data = json.load(f)
                    strategy_data['filepath'] = filepath
                    strategy_data['filename'] = filename
                    strategies.append(strategy_data)
            except Exception as e:
                errors.append(f"Error loading {filename}: {e}")
    
    # Filter by search term
    if search_term:
        search_lower = search_term.lower()
        strategies = [s for s in strategies if 
                     search_lower in s['name'].lower() or 
                     search_lower in ', '.join(s['active_indicators']).lower()]
    
    # Filter by return range
    if min_return is not None:
        strategies = [s for s in strategies if s['performance']['total_return_pct'] >= min_return]
    if max_return is not None:
        strategies = [s for s in strategies if s['performance']['total_return_pct'] <= max_return]
    
    # Sort strategies
    if sort_by == 'return':
        strategies.sort(key=lambda x: x['performance']['total_return_pct'], reverse=not ascending)
    elif sort_by == 'trades':
        strategies.sort(key=lambda x: x['performance']['total_trades'], reverse=not ascending)
    elif sort_by == 'winrate':
        strategies.sort(key=lambda x: x['performance']['win_rate'], reverse=not ascending)
    elif sort_by == 'profit':
        strategies.sort(key=lambda x: x['performance']['total_profit'], reverse=not ascending)
    else:  # timestamp
        strategies.sort(key=lambda x: x['timestamp'], reverse=not ascending)
    
    return strategies, errors

def save_selected_strategies():
    """Save selected strategies based on session state"""
    if 'strategy_data' not in st.session_state:
        st.warning("No strategies available to save!")
        return
    
    saved_count = 0
    for key, strategy_data in st.session_state.strategy_data.items():
        if strategy_data.get('selected', False):
            try:
                filepath = save_strategy_to_storage(strategy_data)
                saved_count += 1
            except Exception as e:
                st.error(f"Error saving strategy {strategy_data['rank']}: {e}")
    
    if saved_count > 0:
        st.success(f"✅ Successfully saved {saved_count} strategy{'ies' if saved_count != 1 else ''}!")
    else:
        st.warning("No strategies were selected for saving.")

def create_implementation_guide(indicators, params, summary):
    """
    Create a practical implementation guide for traders.
    """
    st.write("### 📋 How to Implement This Strategy")
    
    # Strategy overview
    buy_threshold = params.get('buy_score_threshold', 1)
    sell_threshold = params.get('sell_score_threshold', 1)
    
    st.write(f"""
    **Strategy Logic:**
    - This strategy uses a **scoring system** with {len(indicators)} technical indicators
    - **Buy** when at least **{buy_threshold}** indicator(s) signal oversold conditions
    - **Sell** when at least **{sell_threshold}** indicator(s) signal overbought conditions
    - Expected return: **{summary['total_return_pct']:.1f}%** with **{summary['win_rate']:.1f}%** win rate
    """)
    
    # Check for deep learning indicators in the strategy
    dl_indicators = [ind for ind in indicators if ind.startswith('dl_signal_')]
    has_dl_component = len(dl_indicators) > 0
    
    dl_note = ""
    if has_dl_component:
        dl_note = f"- **AI Component:** This strategy uses deep learning pattern recognition"
    
    st.write(f"""
## 📊 **Performance Summary**
- **Backtested Return:** {summary['total_return_pct']:.2f}%
- **Total Trades:** {summary['total_trades']}
- **Win Rate:** {summary['win_rate']:.1f}%
- **Profit Factor:** {summary['profit_factor']:.2f}
- **Max Drawdown:** {summary['max_drawdown_pct']:.2f}%
- **Excess Return vs Buy & Hold:** Consult the metrics above for baseline comparison
{dl_note}
""")

    # Indicator details
    numerical_indicators = [ind for ind in indicators if not ind.startswith(('pattern_', 'dl_signal_'))]
    boolean_indicators = [ind for ind in indicators if ind.startswith(('pattern_', 'dl_signal_'))]
    
    if numerical_indicators:
        st.write("#### 📊 Technical Indicator Rules:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🟢 BUY CONDITIONS (Oversold):**")
            for indicator in numerical_indicators:
                buy_param = params.get(f'{indicator}_buy')
                if buy_param is not None:
                    st.write(f"- {indicator} < {buy_param:.2f}")
        
        with col2:
            st.write("**🔴 SELL CONDITIONS (Overbought):**")
            for indicator in numerical_indicators:
                sell_param = params.get(f'{indicator}_sell')
                if sell_param is not None:
                    st.write(f"- {indicator} > {sell_param:.2f}")
    
    if boolean_indicators:
        st.write("#### 🕯️ Pattern/Signal Rules:")
        for indicator in boolean_indicators:
            buy_use = params.get(f'{indicator}_buy', False)
            sell_use = params.get(f'{indicator}_sell', False)
            
            actions = []
            if buy_use:
                actions.append("BUY")
            if sell_use:
                actions.append("SELL")
            
            if actions:
                st.write(f"- **{indicator}**: Use for {' & '.join(actions)}")
    
    # Implementation steps
    st.write("#### 🛠️ Implementation Steps:")
    
    steps = [
        "**Set up your trading platform** with the required technical indicators",
        f"**Monitor {len(indicators)} indicators** for the conditions listed above",
        f"**Execute BUY** when {buy_threshold}+ indicators show oversold conditions",
        f"**Execute SELL** when {sell_threshold}+ indicators show overbought conditions",
        "**Risk Management**: Set stop-losses at your preferred level (strategy assumes market orders)",
        f"**Expected Performance**: ~{summary['total_trades']/12:.0f} trades per month with {summary['win_rate']:.0f}% success rate"
    ]
    
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")
    
    # Platform recommendations
    st.write("#### 💻 Recommended Platforms:")
    
    platforms = {
        "TradingView": "Advanced charting with all technical indicators + alerts",
        "MetaTrader 4/5": "Automated trading with Expert Advisors (EAs)",  
        "ThinkorSwim": "Professional platform with thinkScript for automation",
        "Interactive Brokers": "API access for algorithmic implementation",
        "Python/QuantConnect": "Full algorithmic trading with backtesting"
    }
    
    for platform, description in platforms.items():
        st.write(f"- **{platform}**: {description}")
    
    # Risk warnings
    st.warning("""
    ⚠️ **Risk Disclaimer**: 
    - Past performance does not guarantee future results
    - Always use proper risk management and position sizing
    - Consider transaction costs and slippage in live trading
    - Test on paper trading first before committing real capital
    """)

def find_bullish_engulfing(data):
    """
    Identifies the Bullish Engulfing pattern.
    """
    return (data['close'].shift(1) < data['open'].shift(1)) & \
           (data['close'] > data['open']) & \
           (data['open'] < data['close'].shift(1)) & \
           (data['close'] > data['open'].shift(1))

def find_bearish_engulfing(data):
    """
    Identifies the Bearish Engulfing pattern.
    """
    return (data['close'].shift(1) > data['open'].shift(1)) & \
           (data['close'] < data['open']) & \
           (data['open'] > data['close'].shift(1)) & \
           (data['close'] < data['open'].shift(1))

def find_doji(data, threshold=0.1):
    """
    Identifies the Doji pattern (open and close are very close).
    """
    body_size = abs(data['open'] - data['close'])
    candle_range = data['high'] - data['low']
    # Use np.where for safe division. If range is 0, it's not a doji.
    return np.where(candle_range > 0, (body_size / candle_range) < threshold, False)

def find_hammer(data):
    """
    Identifies the Hammer pattern (bullish reversal).
    """
    body_size = abs(data['close'] - data['open'])
    is_bullish = data['close'] >= data['open']
    lower_wick = np.where(is_bullish, data['open'] - data['low'], data['close'] - data['low'])
    upper_wick = np.where(is_bullish, data['high'] - data['close'], data['high'] - data['open'])
    candle_range = data['high'] - data['low']
    return (candle_range > 0) & (lower_wick > 2 * body_size) & (upper_wick < body_size)

def find_shooting_star(data):
    """
    Identifies the Shooting Star pattern (bearish reversal).
    """
    body_size = abs(data['close'] - data['open'])
    is_bullish = data['close'] >= data['open']
    upper_wick = np.where(is_bullish, data['high'] - data['close'], data['high'] - data['open'])
    lower_wick = np.where(is_bullish, data['open'] - data['low'], data['close'] - data['low'])
    candle_range = data['high'] - data['low']
    return (candle_range > 0) & (upper_wick > 2 * body_size) & (lower_wick < body_size)

def find_morning_star(data):
    """
    Identifies the Morning Star pattern (bullish reversal).
    """
    first_bearish = data['close'].shift(2) < data['open'].shift(2)
    second_small = abs(data['close'].shift(1) - data['open'].shift(1)) < (data['high'].shift(1) - data['low'].shift(1)) * 0.3
    third_bullish = data['close'] > data['open']
    gap_down = data['open'].shift(1) < data['close'].shift(2)
    gap_up = data['open'] > data['close'].shift(1)
    return first_bearish & second_small & third_bullish & gap_down & gap_up

def find_evening_star(data):
    """
    Identifies the Evening Star pattern (bearish reversal).
    """
    first_bullish = data['close'].shift(2) > data['open'].shift(2)
    second_small = abs(data['close'].shift(1) - data['open'].shift(1)) < (data['high'].shift(1) - data['low'].shift(1)) * 0.3
    third_bearish = data['close'] < data['open']
    gap_up = data['open'].shift(1) > data['close'].shift(2)
    gap_down = data['open'] < data['close'].shift(1)
    return first_bullish & second_small & third_bearish & gap_up & gap_down

from datetime import datetime, timedelta

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_and_validate_data(ticker, period, interval):
    """Load and validate market data from yfinance, with on-disk caching."""
    
    cache_filename = f"{ticker}_{period}_{interval}.csv"
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    if os.path.exists(cache_path):
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < timedelta(days=3):
                logging.info(f"Loading data from cache: {cache_path}")
                cached_data = pd.read_csv(cache_path, parse_dates=['date'])
                # Final validation on cached data
                required = {'open', 'high', 'low', 'close', 'volume', 'date'}
                if required.issubset(cached_data.columns):
                    return cached_data
                else:
                    logging.warning("Cached data is missing columns. Fetching fresh data.")
        except Exception as e:
            logging.error(f"Error reading from cache: {e}. Fetching fresh data.")

    logging.info(f"Cache not found or expired. Fetching data for {ticker} from yfinance.")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"No data for {ticker} from yfinance.")
            return None

        logging.info(f"yfinance returned columns: {data.columns.tolist()}")

        if isinstance(data.columns, pd.MultiIndex):
            # Select the first level of the MultiIndex, which contains OHLCV
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        # Standardize all columns, including the one from reset_index
        data.columns = [str(col).lower() for col in data.columns]

        required = {'open', 'high', 'low', 'close', 'volume', 'date'}
        if not required.issubset(data.columns):
            logging.error(f"Validation failed. Actual columns: {data.columns.tolist()}")
            st.error(f"Missing required columns: {required - set(data.columns)}")
            return None

        data['date'] = pd.to_datetime(data['date'])
        
        logging.info(f"Successfully validated and standardized data for {ticker}.")
        data.to_csv(cache_path, index=False)
        logging.info(f"Saved data to cache: {cache_path}")
        
        return data

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        logging.error(f"Unhandled error in data loading: {e}", exc_info=True)
        return None

# Display saved strategies if requested
if st.session_state.get('show_saved_strategies', False):
    st.write("# 💾 Saved Strategies")
    
    if st.button("⬅️ Back to Main"):
        st.session_state.show_saved_strategies = False
        st.experimental_rerun()
    
    # Search and Filter Controls
    st.write("## 🔍 Search & Filter")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        search_term = st.text_input("🔎 Search by name or indicators", "", key="strategy_search")
    
    with col2:
        sort_options = ['return', 'timestamp', 'trades', 'winrate', 'profit']
        sort_by = st.selectbox("Sort by", sort_options, key="sort_by")
    
    with col3:
        sort_order = st.selectbox("Order", ['Highest First', 'Lowest First'], key="sort_order")
        ascending = sort_order == 'Lowest First'
    
    with col4:
        show_filters = st.checkbox("Advanced Filters", key="show_filters")
    
    # Advanced Filters
    min_return = None
    max_return = None
    if show_filters:
        st.write("### 🎛️ Advanced Filters")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            min_return = st.number_input("Min Return (%)", value=None, key="min_return")
        with filter_col2:
            max_return = st.number_input("Max Return (%)", value=None, key="max_return")
    
    # Load strategies with filters
    saved_strategies, load_errors = load_saved_strategies(
        sort_by=sort_by, 
        ascending=ascending,
        min_return=min_return,
        max_return=max_return,
        search_term=search_term
    )
    
    # Show load errors if any
    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} file(s) failed to load - Click to view errors"):
            for error in load_errors:
                st.error(error)
    
    st.write(f"## 📊 Strategies ({len(saved_strategies)} found)")
    
    if saved_strategies:
        for strategy in saved_strategies:
            return_pct = strategy['performance']['total_return_pct']
            return_icon = "📈" if return_pct > 0 else "📉"
            
            with st.expander(f"{return_icon} {strategy['name']} - {return_pct:.2f}% Return ({len(strategy['active_indicators'])} indicators)"):
                # Performance Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{return_pct:.2f}%", 
                             delta=f"{return_pct - (strategy.get('baseline_return', 0) or 0):.2f}% vs B&H")
                    st.metric("Total Trades", strategy['performance']['total_trades'])
                
                with col2:
                    st.metric("Win Rate", f"{strategy['performance']['win_rate']:.1f}%")
                    st.metric("Profit Factor", f"{strategy['performance']['profit_factor']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{strategy['performance']['max_drawdown_pct']:.2f}%")
                    st.metric("Total Profit", f"${strategy['performance']['total_profit']:,.0f}")
                
                with col4:
                    st.write("**Saved:**")
                    st.write(strategy['timestamp'][:8])
                    st.write("**File:**")
                    st.write(strategy['filename'])
                
                # Indicators
                st.write("**Active Indicators:**")
                indicator_text = ", ".join(strategy['active_indicators'])
                st.write(indicator_text)
                
                # Action buttons
                button_col1, button_col2 = st.columns([1, 4])
                with button_col1:
                    # Use filename for unique key since timestamps might be identical
                    unique_key = strategy['filename'].replace('.json', '').replace('strategy_', '')
                    if st.button(f"🗑️ Delete", key=f"delete_{unique_key}"):
                        os.remove(strategy['filepath'])
                        st.success("Strategy deleted!")
                        st.experimental_rerun()
    else:
        if search_term or min_return is not None or max_return is not None:
            st.info("🔍 No strategies match your search criteria. Try adjusting your filters.")
        else:
            st.info("📁 No saved strategies found. Save some strategies from the main page first!")
    
    st.stop()  # Don't show the rest of the app when viewing saved strategies

# --- User Inputs ---
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Stock Ticker", "SPY").upper()
interval = st.sidebar.selectbox(
    "Select Timeframe",
    ['1d', '5d', '1wk', '1mo', '3mo'],
    index=0
)
period = st.sidebar.selectbox(
    "Select Period",
    ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    index=3
)

# Deep Learning Options
st.sidebar.header("Deep Learning Options")
use_deep_learning = st.sidebar.checkbox("Enable Deep Learning Pattern Detection", value=False)

# Check model availability
if use_deep_learning:
    from real_time_pattern_detector import check_model_availability
    available_models = check_model_availability()
    
    if len(available_models) == 0:
        st.sidebar.warning("⚠️ No trained models found")
        if st.sidebar.button("🤖 Train Deep Learning Models"):
            with st.spinner("Training deep learning models... This may take 30-60 minutes."):
                try:
                    from real_time_pattern_detector import train_pattern_models
                    train_pattern_models()
                    st.sidebar.success("✅ Models trained successfully!")
                except Exception as e:
                    st.sidebar.error(f"❌ Training failed: {e}")
    else:
        st.sidebar.success(f"✅ {len(available_models)}/8 models available")
        
        # Model status
        with st.sidebar.expander("Model Status"):
            from real_time_pattern_detector import CHART_PATTERNS
            for pattern in CHART_PATTERNS:
                status = "✅" if pattern in available_models else "❌"
                st.write(f"{status} {pattern.replace('_', ' ').title()}")

if use_deep_learning:
    st.sidebar.subheader("Pattern Selection")
    
    # Define all available patterns
    rule_based_patterns = ['Bullish Engulfing', 'Bearish Engulfing', 'Doji', 'Hammer']
    complex_patterns = ['Head and Shoulders', 'Inverse Head and Shoulders', 'Double Top', 'Double Bottom', 'Cup and Handle', 'Triangle', 'Flag', 'Pennant']
    
    # Group patterns by type
    pattern_groups = {
        "Rule-Based Candlestick Patterns": rule_based_patterns,
        "Complex Chart Patterns": complex_patterns
    }
    
    # Create a dictionary to store selected patterns
    selected_patterns = {}
    
    # Allow selection of multiple patterns
    patterns_to_detect = st.sidebar.multiselect(
        "Select patterns to detect:",
        rule_based_patterns + complex_patterns,
        default=["Bullish Engulfing"]
    )
    
    if any(pattern in complex_patterns for pattern in patterns_to_detect):
        st.sidebar.info("Complex patterns use synthetic data for training. The system will automatically generate this data when needed.")
    
    training_mode = st.sidebar.checkbox("Training Mode", value=False, 
                                      help="Enable to train a new model using rule-based patterns as training data")
    
    if training_mode:
        num_samples = st.sidebar.slider("Number of Training Samples", min_value=10, max_value=100, value=30)
        epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=20, value=5)

if st.sidebar.button("Find Patterns"):
    st.write(f"Searching for patterns in {ticker} on the {interval} timeframe...")

    try:
        logging.info(f"Fetching data for {ticker} with period={period} and interval={interval}")
        data = yf.download(ticker, period=period, interval=interval)
        
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the inputs.")
            logging.warning(f"No data returned for ticker {ticker}")
        else:
            st.success(f"Successfully downloaded data for {ticker}.")
            logging.info(f"Downloaded {len(data)} rows of data.")

            # Flatten the multi-level column headers from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(-1)
            data.columns = pd.Index([str(c).lower() for c in data.columns])
            data = data.reset_index()
            data.columns = pd.Index([str(c).lower() for c in data.columns])
            logging.info("DataFrame head after reset_index and column flattening:")
            logging.info(data.head())

            # --- Pattern Recognition (on a copy to prevent data corruption) ---
            try:
                logging.info("Running pattern detection functions.")
                analysis_data = data.copy()
                patterns = {
                    'Bullish Engulfing': find_bullish_engulfing(analysis_data),
                    'Bearish Engulfing': find_bearish_engulfing(analysis_data),
                    'Doji': find_doji(analysis_data),
                    'Hammer': find_hammer(analysis_data)
                }
                logging.info("Successfully ran all pattern detection functions.")
            except Exception as e:
                logging.error(f"Error during pattern detection: {e}", exc_info=True)
                st.error(f"An error occurred during pattern detection: {e}")
                patterns = {}

            # --- Chart Display ---
            logging.info("Creating candlestick chart.")
            try:
                fig = go.Figure(data=[go.Candlestick(
                    x=data['date'],
                    open=data['open'], 
                    high=data['high'],
                    low=data['low'], 
                    close=data['close'],
                    name='Candlestick'
                )])
                
                # Add pattern markers
                pattern_markers = {
                    'Bullish Engulfing': {'symbol': 'triangle-up', 'color': 'lime', 'position_col': 'low', 'multiplier': 0.98},
                    'Bearish Engulfing': {'symbol': 'triangle-down', 'color': 'red', 'position_col': 'high', 'multiplier': 1.02},
                    'Doji': {'symbol': 'diamond', 'color': 'yellow', 'position_col': 'high', 'multiplier': 1.02},
                    'Hammer': {'symbol': 'triangle-up', 'color': 'cyan', 'position_col': 'low', 'multiplier': 0.98}
                }
                
                for name, series in patterns.items():
                    if series.dtype == 'bool' and series.any():
                        pattern_days = data[series]
                        fig.add_trace(go.Scatter(
                            x=pattern_days['date'],
                            y=pattern_days[pattern_markers[name]['position_col']] * pattern_markers[name]['multiplier'],
                            mode='markers',
                            marker=dict(
                                symbol=pattern_markers[name]['symbol'],
                                color=pattern_markers[name]['color'],
                                size=10),
                            name=name
                        ))
                
                # Deep Learning Pattern Detection
                if use_deep_learning and patterns_to_detect:
                    st.subheader("Deep Learning Pattern Detection")
                    
                    # Define markers for complex patterns
                    complex_pattern_markers = {
                        'Head and Shoulders': {'symbol': 'star', 'color': 'magenta', 'position_col': 'high', 'multiplier': 1.02},
                        'Inverse Head and Shoulders': {'symbol': 'star', 'color': 'cyan', 'position_col': 'low', 'multiplier': 0.98},
                        'Double Top': {'symbol': 'circle', 'color': 'red', 'position_col': 'high', 'multiplier': 1.02},
                        'Double Bottom': {'symbol': 'circle', 'color': 'green', 'position_col': 'low', 'multiplier': 0.98},
                        'Cup and Handle': {'symbol': 'square', 'color': 'orange', 'position_col': 'low', 'multiplier': 0.98},
                        'Triangle': {'symbol': 'triangle-up', 'color': 'purple', 'position_col': 'close', 'multiplier': 1.01},
                        'Flag': {'symbol': 'pentagon', 'color': 'blue', 'position_col': 'close', 'multiplier': 1.01},
                        'Pennant': {'symbol': 'hexagon', 'color': 'pink', 'position_col': 'close', 'multiplier': 1.01}
                    }
                    
                    # Combine all pattern markers
                    all_pattern_markers = {**pattern_markers, **complex_pattern_markers}
                    
                    # Process each selected pattern
                    for pattern_name in patterns_to_detect:
                        with st.spinner(f"Processing {pattern_name} pattern..."):
                            st.write(f"**{pattern_name}:**")
                            
                            if training_mode:
                                # Training mode: use rule-based patterns as ground truth for rule-based patterns
                                # For complex patterns, use synthetic data
                                
                                pattern_indices = []
                                non_pattern_indices = []
                                
                                if pattern_name in rule_based_patterns:
                                    # For rule-based patterns, use the rule-based detection as ground truth
                                    if pattern_name in patterns:
                                        pattern_indices = data[patterns[pattern_name]].index.tolist()
                                        st.write(f"Found {len(pattern_indices)} {pattern_name} patterns for training.")
                                else:
                                    # For complex patterns, use synthetic data
                                    st.info(f"Using synthetic data for training '{pattern_name}' pattern.")
                                    X_paths, y_train = use_synthetic_data_for_training(pattern_name, num_samples=50)
                                    
                                    if X_paths:
                                        st.success(f"Generated/found {len(X_paths)} synthetic images for {pattern_name} pattern.")
                                        
                                        # Load and preprocess images
                                        X_train = []
                                        valid_indices = []
                                        
                                        for i, path in enumerate(X_paths):
                                            img = load_and_preprocess_image(path)
                                            if img is not None:
                                                X_train.append(img)
                                                valid_indices.append(i)
                                        
                                        if not X_train:
                                            st.error(f"Failed to load training images for {pattern_name}.")
                                            continue
                                        
                                        # Convert to numpy arrays
                                        X_train = np.array(X_train)
                                        y_train = np.array([y_train[i] for i in valid_indices])
                                        
                                        # Build and train model
                                        model = build_cnn_model(input_shape=X_train[0].shape)
                                        
                                        # Display training progress
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        # Train the model
                                        for epoch in range(epochs):
                                            model.fit(X_train, y_train, epochs=1, batch_size=8, verbose=0)
                                            progress_bar.progress((epoch + 1) / epochs)
                                            status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                                        
                                        # Save the model
                                        model_dir = 'models'
                                        os.makedirs(model_dir, exist_ok=True)
                                        model_path = f"{model_dir}/{pattern_name.lower().replace(' ', '_')}_model.h5"
                                        model.save(model_path)
                                        
                                        st.success(f"Model trained and saved to {model_path}")
                                        continue
                                
                                if not pattern_indices:
                                    st.warning(f"No {pattern_name} patterns found for training. Please select a different pattern or timeframe.")
                                    continue
                            
                                if pattern_name in rule_based_patterns and pattern_indices:
                                    # Get indices where the pattern doesn't appear
                                    non_pattern_indices = data[~patterns[pattern_name]].index.tolist()
                                    
                                    # Limit the number of samples
                                    if len(pattern_indices) > num_samples // 2:
                                        pattern_indices = random.sample(pattern_indices, num_samples // 2)
                                    
                                    if len(non_pattern_indices) > num_samples // 2:
                                        non_pattern_indices = random.sample(non_pattern_indices, num_samples // 2)
                                    
                                    # Generate training data
                                    X_paths, y_train = generate_training_data(
                                        data, 
                                        pattern_indices, 
                                        non_pattern_indices, 
                                        window_size=20
                                    )
                            
                                    if not X_paths:
                                        st.error(f"Failed to generate training data for {pattern_name}.")
                                        continue
                                    
                                    # Load and preprocess images
                                    X_train = []
                                    valid_indices = []
                                    for i, path in enumerate(X_paths):
                                        img = load_and_preprocess_image(path)
                                        if img is not None:
                                            X_train.append(img)
                                            valid_indices.append(i)
                                    
                                    if not X_train:
                                        st.error(f"Failed to load training images for {pattern_name}.")
                                        continue
                                    
                                    # Convert to numpy arrays
                                    X_train = np.array(X_train)
                                    y_train = np.array([y_train[i] for i in valid_indices])
                                    
                                    # Build and train model
                                    model = build_cnn_model(input_shape=X_train[0].shape)
                                    
                                    # Display training progress
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Train the model
                                    for epoch in range(epochs):
                                        model.fit(X_train, y_train, epochs=1, batch_size=8, verbose=0)
                                        progress_bar.progress((epoch + 1) / epochs)
                                        status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                                    
                                    # Save the model
                                    model_dir = 'models'
                                    os.makedirs(model_dir, exist_ok=True)
                                    model_path = f"{model_dir}/{pattern_name.lower().replace(' ', '_')}_model.h5"
                                    model.save(model_path)
                                    
                                    st.success(f"Model trained and saved to {model_path}")
                            
                            else:
                                # Inference mode: use pre-trained model to detect patterns
                                model_dir = 'models'
                                model_path = f"{model_dir}/{pattern_name.lower().replace(' ', '_')}_model.h5"
                                
                                if not os.path.exists(model_path):
                                    st.warning(f"No trained model found for {pattern_name}. Please train a model first.")
                                    continue
                                
                                try:
                                    # Load the model
                                    model = tf.keras.models.load_model(model_path)
                                    st.success(f"Loaded model from {model_path}")
                                    
                                    # Generate images for each candle and predict
                                    dl_pattern_indices = []
                                    
                                    # For demonstration, only check the last 30 candles
                                    check_indices = list(range(max(0, len(data) - 30), len(data)))
                                    
                                    for idx in check_indices:
                                        # Generate chart image
                                        image_path = create_chart_image(
                                            data, 
                                            idx, 
                                            window_size=20, 
                                            save_path=f"chart_images/check_{idx}.png"
                                        )
                                        
                                        if image_path:
                                            # Load and preprocess image
                                            img = load_and_preprocess_image(image_path)
                                            if img is not None:
                                                # Make prediction
                                                prediction = model.predict(np.array([img]), verbose=0)[0][0]
                                                if prediction > 0.7:  # Threshold for positive detection
                                                    dl_pattern_indices.append(idx)
                                    
                                    # Add deep learning pattern markers to the chart
                                    if dl_pattern_indices:
                                        dl_pattern_days = data.iloc[dl_pattern_indices]
                                        
                                        # Determine marker style based on pattern type
                                        marker_style = {
                                            'symbol': 'star',
                                            'color': 'purple',
                                            'size': 12
                                        }
                                        
                                        # Use specific marker styles for complex patterns
                                        if pattern_name in complex_pattern_markers:
                                            marker_style['symbol'] = complex_pattern_markers[pattern_name]['symbol']
                                            marker_style['color'] = complex_pattern_markers[pattern_name]['color']
                                            position_col = complex_pattern_markers[pattern_name]['position_col']
                                            multiplier = complex_pattern_markers[pattern_name]['multiplier']
                                        else:
                                            # Default for rule-based patterns
                                            position_col = 'high'
                                            multiplier = 1.03
                                        
                                        fig.add_trace(go.Scatter(
                                            x=dl_pattern_days['date'],
                                            y=dl_pattern_days[position_col] * multiplier,
                                            mode='markers',
                                            marker=dict(
                                                symbol=marker_style['symbol'],
                                                color=marker_style['color'],
                                                size=marker_style['size']),
                                            name=f"DL {pattern_name}"
                                        ))
                                        st.write(f"Deep learning detected {len(dl_pattern_indices)} {pattern_name} patterns.")
                                    else:
                                        st.info(f"No {pattern_name} patterns detected by deep learning.")
                                        
                                except Exception as e:
                                    logging.error(f"Error using deep learning model: {e}", exc_info=True)
                                    st.error(f"An error occurred while using the deep learning model: {e}")
                
                fig.update_layout(
                    title=f'{ticker} Candlestick Chart - {interval} timeframe',
                    yaxis_title='Price (USD)',
                    xaxis_title='Date',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                logging.info("Chart rendered successfully.")
                
            except Exception as e:
                logging.error(f"Error rendering chart: {e}", exc_info=True)
                st.error(f"An error occurred while rendering the chart: {e}")
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Enter a stock ticker and select a timeframe to begin.")

# --- Saved Strategies Viewer ---
st.sidebar.header("💾 Saved Strategies")

saved_strategies, load_errors = load_saved_strategies()

if load_errors:
    st.sidebar.error(f"⚠️ {len(load_errors)} file(s) failed to load")
    with st.sidebar.expander("View Errors"):
        for error in load_errors:
            st.sidebar.error(error)

if saved_strategies:
    st.sidebar.write(f"📊 **{len(saved_strategies)} saved strategies**")
    
    if st.sidebar.button("📋 View All Saved Strategies"):
        st.session_state.show_saved_strategies = True
    
    # Quick preview of top strategy by return
    top_strategy = max(saved_strategies, key=lambda x: x['performance']['total_return_pct'])
    st.sidebar.write(f"**Best:** {top_strategy['name'][:25]}...")
    st.sidebar.write(f"Return: {top_strategy['performance']['total_return_pct']:.1f}%")
    st.sidebar.write(f"Indicators: {len(top_strategy['active_indicators'])}")
else:
    st.sidebar.write("No saved strategies yet")

st.sidebar.markdown("---")

# --- Automated Strategy Optimization ---
st.sidebar.header("Strategy Discovery Engine")

# Performance Settings in Sidebar (before optimization starts)
st.sidebar.subheader("⚡ Performance Settings")

import multiprocessing as mp
max_cores = mp.cpu_count()

# Parallel workers selection
n_jobs = st.sidebar.selectbox(
    "Parallel Workers", 
    options=[1, 2, 4, max_cores-1, max_cores],
    index=3,  # Default to max_cores-1 (all except 1)
    help=f"Your system has {max_cores} CPU cores. Recommended: {max_cores-1} (leaves 1 core free)"
)

# Optimization method selection  
optimization_method = st.sidebar.selectbox(
    "Optimization Method",
    options=['standard', 'joblib', 'advanced'],
    index=1,  # Default to joblib (often fastest)
    help="Joblib is usually fastest. Advanced uses distributed processing."
)

# Show estimated time
estimated_time = (5000 / n_jobs) / 60
speedup_text = f"{n_jobs}x speedup" if n_jobs > 1 else "single-core"
st.sidebar.metric("Estimated Time", f"{estimated_time:.1f} min", speedup_text)

# Starting Capital
st.sidebar.subheader("💰 Portfolio Settings")
starting_capital = st.sidebar.number_input(
    "Starting Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000,
    help="Initial capital for backtesting strategies"
)

# Benchmark option in sidebar
run_benchmark = st.sidebar.checkbox(
    "🏁 Run Benchmark First", 
    help="Test methods to find fastest (adds 2-3 min but optimizes the full run)"
)

if st.sidebar.button("Find and Optimize Top Strategies"):
    st.subheader("Optimized Strategy Results")
    
    # Clear previous results
    if 'optimization_results' in st.session_state:
        del st.session_state.optimization_results
    if 'strategy_data' in st.session_state:
        del st.session_state.strategy_data

    try:
        data = load_and_validate_data(ticker, period, interval)
        if data is not None:
            with st.spinner("Calculating indicators and patterns..."):
                # Add candlestick patterns as boolean columns
                data['pattern_bullish_engulfing'] = find_bullish_engulfing(data)
                data['pattern_bearish_engulfing'] = find_bearish_engulfing(data)
                data['pattern_hammer'] = find_hammer(data)
                data['pattern_doji'] = find_doji(data)

                # Add deep learning signals (if enabled)
                if use_deep_learning:
                    st.info("🤖 Running deep learning pattern detection...")
                    try:
                        from real_time_pattern_detector import RealTimePatternDetector
                        
                        # Initialize pattern detector
                        detector = RealTimePatternDetector(load_models=True)
                        
                        if detector.models_loaded:
                            # Get DL-based buy/sell signals
                            dl_buy_signals, dl_sell_signals = detector.get_pattern_based_signals(
                                data, confidence_threshold=0.7
                            )
                            
                            data['dl_signal_buy'] = dl_buy_signals
                            data['dl_signal_sell'] = dl_sell_signals
                            
                            # Show detected patterns
                            patterns = detector.detect_patterns_in_data(data, confidence_threshold=0.7)
                            if patterns:
                                st.success(f"🎯 Detected {len(patterns)} chart patterns!")
                                for pattern_name, pattern_data in patterns.items():
                                    confidence = pattern_data['probability'] * 100
                                    pattern_type = pattern_data['pattern_type']
                                    
                                    # Use appropriate emoji for pattern type
                                    emoji = "📈" if pattern_type == "bullish" else "📉" if pattern_type == "bearish" else "🔄"
                                    
                                    st.write(f"{emoji} **{pattern_name.replace('_', ' ').title()}** - "
                                           f"{pattern_type.title()} - {confidence:.1f}% confidence")
                            else:
                                st.info("No significant patterns detected in recent data.")
                        else:
                            st.warning("⚠️ Deep learning models not available. Please train models first.")
                            # Fallback to simple signals
                            data['dl_signal_buy'] = pd.Series(False, index=data.index)
                            data['dl_signal_sell'] = pd.Series(False, index=data.index)
                            
                    except Exception as e:
                        st.error(f"❌ Deep learning error: {e}")
                        # Fallback to simple signals
                        data['dl_signal_buy'] = pd.Series(False, index=data.index)
                        data['dl_signal_sell'] = pd.Series(False, index=data.index)

                enriched_data = get_all_indicators(data)
            st.success("Indicator and pattern calculation complete!")

            # Run benchmark if requested
            if run_benchmark:
                st.write("**🏁 Performance Benchmark:**")
                with st.spinner("Benchmarking optimization methods..."):
                    try:
                        from advanced_optimization import benchmark_optimization_methods
                        results, best_method = benchmark_optimization_methods(enriched_data, n_trials=100)
                        
                        st.success(f"🏆 Best method for your system: **{best_method}**")
                        
                        # Show results table
                        benchmark_data = []
                        for method, data in results.items():
                            if data['success']:
                                benchmark_data.append({
                                    'Method': method,
                                    'Duration (s)': f"{data['duration']:.1f}",
                                    'Trials/sec': f"{data['trials_per_second']:.1f}",
                                    'Best Return': f"{data['best_return']:.2f}%"
                                })
                        
                        if benchmark_data:
                            st.table(benchmark_data)
                            
                            # Auto-update optimization method if benchmark found a better one
                            if best_method != optimization_method:
                                st.info(f"💡 Recommendation: **{best_method}** is faster than **{optimization_method}** on your system")
                                
                    except Exception as e:
                        st.error(f"Benchmark failed: {e}")
                
                st.markdown("---")

            # FORCE DEBUG - ALWAYS PRINT THIS
            print(f"\n🚨🚨🚨 OPTIMIZATION ENTRY POINT - METHOD: {optimization_method} 🚨🚨🚨")
            print(f"🚨🚨🚨 FORCING STANDARD OPTIMIZATION FOR DEBUGGING 🚨🚨🚨\n")
            
            with st.spinner(f"Running {optimization_method} optimization with {n_jobs} parallel workers... This may take several minutes."):
                # Temporarily force standard optimization for debugging
                if True:  # Force standard for now
                    st.info("🔍 Using standard optimization for debugging")
                    print("🔥🔥🔥 CALLING run_optimization FROM optimization.py 🔥🔥🔥")
                    top_trials = run_optimization(enriched_data, n_trials=10000, n_jobs=n_jobs)  # Use parallel processing
                else:
                    if optimization_method == 'standard':
                        top_trials = run_optimization(enriched_data, n_trials=5000, n_jobs=n_jobs)
                    else:
                        try:
                            from advanced_optimization import run_optimization_distributed
                            method_map = {'joblib': 'joblib', 'advanced': 'distributed'}
                            top_trials = run_optimization_distributed(
                                enriched_data, 
                                n_trials=5000, 
                                n_jobs=n_jobs,
                                method=method_map[optimization_method]
                            )
                        except ImportError:
                            st.warning("Advanced optimization not available, using standard method")
                            top_trials = run_optimization(enriched_data, n_trials=5000, n_jobs=n_jobs)
            
            # Store optimization results in session state
            st.session_state.optimization_results = {
                'top_trials': top_trials,
                'enriched_data': enriched_data,
                'ticker': ticker,
                'period': period,
                'interval': interval
            }
            
            st.success("Optimization complete!")

    except Exception as e:
        logging.error(f"Error during strategy optimization: {e}", exc_info=True)
        st.error(f"An error occurred during strategy optimization: {e}")

# Display optimization results if they exist in session state
if 'optimization_results' in st.session_state:
    results = st.session_state.optimization_results
    top_trials = results['top_trials']
    enriched_data = results['enriched_data']
    
    st.subheader("📊 Strategy Results")
    
    st.write("**Top 3 Discovered Strategies:**")
    
    # Process each of the top 3 strategies
    for rank, trial in enumerate(top_trials, 1):
        if trial.value is None or trial.value <= -1e9:
            continue
            
        # Extract strategy details
        active_indicators = [param.replace('use_', '') for param in trial.params.keys() 
                           if param.startswith('use_') and trial.params[param] == True]
        strategy_name = f"Strategy {rank}: " + ' + '.join(active_indicators[:4]) + ('...' if len(active_indicators) > 4 else '')
        
        # Add checkbox for strategy selection
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            selected = st.checkbox("Save", key=f"save_strategy_{rank}", help="Select to save this strategy", label_visibility="hidden")
        with col2:
            with st.expander(f"🏆 Rank #{rank}: {trial.value:.2f}% Return - {len(active_indicators)} Indicators"):
                
                # Strategy Configuration - Full Width
                st.write("**📊 Strategy Configuration:**")
                config_col1, config_col2, config_col3, config_col4 = st.columns([1, 1, 1, 2])
                
                with config_col1:
                    st.metric("Active Indicators", len(active_indicators))
                with config_col2:
                    st.metric("Buy Threshold", trial.params.get('buy_score_threshold', 1))
                with config_col3:
                    st.metric("Sell Threshold", trial.params.get('sell_score_threshold', 1))
                with config_col4:
                    # Show indicator categories
                    numerical_indicators = [ind for ind in active_indicators if not ind.startswith(('pattern_', 'dl_signal_'))]
                    boolean_indicators = [ind for ind in active_indicators if ind.startswith(('pattern_', 'dl_signal_'))]
                    
                    if numerical_indicators:
                        st.write(f"**Technical:** {', '.join(numerical_indicators[:4])}" + 
                               (f" +{len(numerical_indicators)-4} more" if len(numerical_indicators) > 4 else ""))
                    if boolean_indicators:
                        st.write(f"**Patterns:** {', '.join(boolean_indicators)}")
                        
                
                st.markdown("---")
                
                # Re-run the backtest to get detailed results
                from optimization import universal_strategy
                
                final_params = trial.params.copy()
                # Ensure score thresholds are present (they should be now, but safety check)
                if 'buy_score_threshold' not in final_params:
                    final_params['buy_score_threshold'] = 1
                if 'sell_score_threshold' not in final_params:
                    final_params['sell_score_threshold'] = 1
                
                backtester = Backtester(enriched_data, strategy_name, universal_strategy, final_params, starting_capital)
                backtester.run()
                trade_log, summary = backtester.get_results()
                
                # Store strategy data for potential saving
                if 'strategy_data' not in st.session_state:
                    st.session_state.strategy_data = {}
                
                st.session_state.strategy_data[f"strategy_{rank}"] = {
                    'selected': selected,
                    'rank': rank,
                    'name': strategy_name,
                    'trial': trial,
                    'summary': summary,
                    'trade_log': trade_log,
                    'active_indicators': active_indicators,
                    'baseline': None  # Will be updated below
                }
                
                # Calculate buy-and-hold baseline for comparison  
                baseline = calculate_buy_and_hold_baseline(enriched_data, starting_capital)
                excess_return = summary['total_return_pct'] - baseline['total_return_pct']
                
                # Update baseline in stored strategy data
                if f"strategy_{rank}" in st.session_state.strategy_data:
                    st.session_state.strategy_data[f"strategy_{rank}"]["baseline"] = baseline

                st.write("**📈 Performance Metrics vs. Buy & Hold Baseline:**")
                
                # Show baseline comparison first
                baseline_col1, baseline_col2, baseline_col3 = st.columns([1, 1, 1])
                with baseline_col1:
                    st.metric("Strategy Return", f"{summary['total_return_pct']:.2f}%")
                with baseline_col2:
                    st.metric("Buy & Hold Return", f"{baseline['total_return_pct']:.2f}%")
                with baseline_col3:
                    color = "normal" if excess_return >= 0 else "inverse"
                    st.metric("Excess Return", f"{excess_return:+.2f}%", 
                            delta=f"{excess_return:+.2f}%", delta_color=color)
                
                st.markdown("---")
                
                if summary['total_trades'] > 0:
                    # Create metrics columns - use more space
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Total Return", f"{summary['total_return_pct']:.2f}%")
                        st.metric("Total Trades", summary['total_trades'])
                    
                    with metric_col2:
                        st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
                        st.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
                    
                    with metric_col3:
                        st.metric("Max Drawdown", f"{summary['max_drawdown_pct']:.2f}%")
                        st.metric("Ending Capital", f"${summary['ending_capital']:,.0f}")
                    
                    with metric_col4:
                        avg_trade = summary['total_profit'] / summary['total_trades'] if summary['total_trades'] > 0 else 0
                        st.metric("Avg Trade", f"${avg_trade:.0f}")
                        st.metric("Total Profit", f"${summary['total_profit']:,.0f}")

                # Generate signals for visualization
                signals = universal_strategy(enriched_data, final_params)
                
                # Create interactive chart
                st.write("**📊 Strategy Performance Visualization:**")
                
                # Create the interactive plot (simplified - only showing trade entries/exits)
                fig = create_strategy_chart(enriched_data, trade_log, strategy_name, summary, baseline)
                st.plotly_chart(fig, use_container_width=True)
                
                # Implementation Guide
                st.write("**🚀 Strategy Implementation Guide:**")
                create_implementation_guide(active_indicators, trial.params, summary)
                
                # Create tabs for additional details
                tab1, tab2 = st.tabs(["📋 Trade Details", "🔧 Parameters"])
                
                with tab1:
                    if summary['total_trades'] > 0:
                        # Add cumulative profit column to trade log
                        trade_log_display = trade_log.copy()
                        trade_log_display['total_profit'] = trade_log_display['profit'].cumsum()
                        
                        # Rename columns for clarity
                        trade_log_display = trade_log_display.rename(columns={
                            'entry_date': 'Entry Date',
                            'exit_date': 'Exit Date', 
                            'entry_price': 'Entry Price',
                            'exit_price': 'Exit Price',
                            'contracts': 'Shares/Contracts',
                            'position_value': 'Position Size ($)',
                            'profit': 'Trade P&L',
                            'total_profit': 'Running Total'
                        })
                        
                        st.write(f"**Trade Log ({len(trade_log)} trades):**")
                        st.caption("💡 **Position Size** = Total dollars invested in each trade (all available capital)")
                        st.dataframe(trade_log_display.style.format({
                            'Entry Price': '${:.2f}',
                            'Exit Price': '${:.2f}',
                            'Shares/Contracts': '{:.4f}',
                            'Position Size ($)': '${:,.2f}',
                            'Trade P&L': '${:.2f}',
                            'Running Total': '${:.2f}'
                        }), use_container_width=True)
                    else:
                        st.info("No trades were executed with these parameters.")
                
                with tab2:
                    st.write("**Detailed Parameters:**")
                    st.json(trial.params)
                
                st.markdown("---")  # Separator between strategies

    # Strategy storage section (moved outside the loop)
    st.markdown("---")
    st.write("## 💾 Strategy Storage")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Select strategies above and click save to store them permanently:")
        if 'strategy_data' in st.session_state:
            selected_count = sum(1 for data in st.session_state.strategy_data.values() if data.get('selected', False))
            st.write(f"**{selected_count} strategies selected**")
    with col2:
        if st.button("💾 Save Selected", type="primary", key="save_strategies_btn"):
            save_selected_strategies()
