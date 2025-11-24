import os
import warnings
import logging

# CRITICAL: Universal TensorFlow configuration for all machines
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only on ALL machines
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs completely
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent library conflicts
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU memory growth
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can hang

# Aggressively suppress ALL Streamlit runtime warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# CRITICAL: Global flag to prevent duplicate import messages across ALL modules
os.environ['PATTERN_FINDR_IMPORTS_LOGGED'] = 'false'

# Suppress all warnings that clutter console - MAXIMUM SUPPRESSION
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.state").setLevel(logging.CRITICAL)

# Additional warning suppression (without accessing non-existent attributes)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.CRITICAL)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import random

# Print BEFORE any TensorFlow imports to prevent module import loops
print("🔧 TensorFlow CPU-only mode enabled for universal compatibility")

# Import TensorFlow with universal CPU-only configuration
import tensorflow as tf

# Immediately configure TensorFlow for CPU-only operation
tf.config.set_visible_devices([], 'GPU')

# Import deep learning modules (these may also import TensorFlow)
from dl_pattern_detector import create_chart_image, generate_training_data, build_cnn_model, load_and_preprocess_image, use_synthetic_data_for_training

HAS_TENSORFLOW = True
from backtester import Backtester
from optimization import run_optimization
from indicators import get_all_indicators

def save_portfolio_config():
    """Save current portfolio configuration for model training sync"""
    try:
        config = {
            'tickers': st.session_state.get('portfolio_tickers', []),
            'saved_by': 'streamlit_app',
            'timestamp': datetime.now().isoformat()
        }
        with open('portfolio_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass  # Silent fail - not critical

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
    baseline_vals = []
    if baseline is not None:
        # Calculate daily buy-and-hold value
        baseline_shares = baseline['starting_capital'] / data.iloc[0]['close']
        baseline_daily_value = [baseline_shares * price for price in data['close']]
        baseline_vals = baseline_daily_value
        
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
        height=1200,  # Increased height for better visibility
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
        margin=dict(l=80, r=80, t=120, b=100),  # Increased margins
        hovermode='x unified',  # Better hover interaction
        xaxis=dict(
            rangeslider=dict(visible=False)  # DISABLE THE RANGE SLIDER
        )
    )
    
    # Force x-axis to show full data range (prevent auto-zoom to trades only)
    if len(data) > 0:
        fig.update_xaxes(
            range=[data.index[0], data.index[-1]],
            row=1, col=1
        )
        fig.update_xaxes(
            range=[data.index[0], data.index[-1]],
            row=2, col=1
        )
    
    # Format the portfolio value y-axis with better scaling
    # Incorporate baseline values into range calculation
    all_values = portfolio_value + baseline_vals
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Use more generous padding for better visibility
        value_range = max_val - min_val
        if value_range > 0:
            padding = value_range * 0.20  # 20% padding for better visibility
        else:
            padding = max_val * 0.05  # 5% of max value if flat line
        
        fig.update_yaxes(
            title_text="Portfolio Value ($)",
            tickformat='$,.0f',
            range=[max(0, min_val - padding), max_val + padding],  # Don't go below $0
            nticks=10,  # More tick marks for better readability
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

def create_30day_performance_chart(ticker, price_data, trade_log, strategy_name, return_pct):
    """Create a compact chart showing 30-day performance with trades"""
    if price_data is None or len(price_data) == 0:
        return None
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['close'],
        mode='lines',
        name='Price',
        line=dict(color='#2962FF', width=2),
        hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add buy and sell markers from trade log
    # Trade log has: entry_date, exit_date, entry_price, exit_price
    if len(trade_log) > 0 and 'entry_date' in trade_log.columns:
        # Add entry (buy) markers
        fig.add_trace(go.Scatter(
            x=trade_log['entry_date'],
            y=trade_log['entry_price'],
            mode='markers',
            name='Buy',
            marker=dict(symbol='triangle-up', size=12, color='#00C853'),
            hovertemplate='BUY<br>%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add exit (sell) markers
        fig.add_trace(go.Scatter(
            x=trade_log['exit_date'],
            y=trade_log['exit_price'],
            mode='markers',
            name='Sell',
            marker=dict(symbol='triangle-down', size=12, color='#FF1744'),
            hovertemplate='SELL<br>%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title=f"{ticker} - Last 30 Days Performance: {return_pct:+.1f}%",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60),
        template='plotly_dark'
    )
    
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
        'ticker': strategy_data.get('ticker', 'Unknown'),
        'period': strategy_data.get('period', '1y'),
        'interval': strategy_data.get('interval', '1d'),
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

# Conditionally apply cache decorator only when in Streamlit runtime context
def _cache_if_streamlit(func):
    """Apply Streamlit cache decorator only if in runtime context"""
    try:
        # Test if we're in Streamlit runtime
        if hasattr(st, 'runtime') and st.runtime.exists():
            return st.cache_data(ttl=300)(func)
        else:
            return func
    except:
        return func

@_cache_if_streamlit
def load_saved_strategies(sort_by='timestamp', ascending=False, min_return=None, max_return=None, search_term=''):
    """Load all saved strategies from storage with filtering and sorting options"""
    storage_dir = "saved_strategies"
    if not os.path.exists(storage_dir):
        return [], []
    
    strategies = []
    errors = []
    seen_timestamps = set()  # Track unique strategies by timestamp
    
    for filename in os.listdir(storage_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(storage_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    strategy_data = json.load(f)
                    
                    # Skip duplicates based on timestamp
                    timestamp = strategy_data.get('timestamp')
                    if timestamp in seen_timestamps:
                        continue
                    seen_timestamps.add(timestamp)
                    
                    strategy_data['filepath'] = filepath
                    strategy_data['filename'] = filename
                    # Ensure ticker field exists for older strategies
                    if 'ticker' not in strategy_data:
                        strategy_data['ticker'] = 'Unknown'
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
    
    # Determine cache expiry based on interval
    if interval in ['1m', '5m']:
        cache_expiry = timedelta(minutes=30)  # Refresh intraday data frequently
    elif interval in ['15m', '30m']:
        cache_expiry = timedelta(hours=2)
    elif interval in ['1h']:
        cache_expiry = timedelta(hours=6)
    elif interval in ['1d']:
        cache_expiry = timedelta(days=1)  # Daily data refreshes daily
    else:
        cache_expiry = timedelta(days=3)  # Weekly/monthly can be cached longer
    
    if os.path.exists(cache_path):
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < cache_expiry:
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
        
        # Handle both 'date' and 'datetime' column names (depends on interval)
        if 'datetime' in data.columns and 'date' not in data.columns:
            data.rename(columns={'datetime': 'date'}, inplace=True)

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

# Display Trade section if requested
if st.session_state.get('show_trade_section', False):
    st.write("# 🎯 Trade - Daily Portfolio Signals")
    
    if st.button("⬅️ Back to Main"):
        st.session_state.show_trade_section = False
        st.rerun()
    
    st.markdown("---")
    
    # Initialize portfolio in session state if not exists
    if 'portfolio_tickers' not in st.session_state:
        # Try to load from saved portfolio config
        try:
            if os.path.exists('portfolio_config.json'):
                with open('portfolio_config.json', 'r') as f:
                    portfolio_data = json.load(f)
                    st.session_state.portfolio_tickers = portfolio_data.get('tickers', ['SPY', 'MSTY', 'MSTR'])
            else:
                st.session_state.portfolio_tickers = ['SPY', 'MSTY', 'MSTR']
        except Exception:
            st.session_state.portfolio_tickers = ['SPY', 'MSTY', 'MSTR']
    
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = []
    
    if 'alert_settings' not in st.session_state:
        st.session_state.alert_settings = {
            'sms_enabled': False,
            'phone_number': '',
            'twilio_account_sid': '',
            'twilio_auth_token': '',
            'twilio_phone_number': '',
            'email_alerts': False,
            'email_address': ''
        }
    
    # Portfolio Management Section
    st.subheader("📊 Portfolio Management")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Current Portfolio:**")
        portfolio_display = ", ".join(st.session_state.portfolio_tickers)
        st.info(f"🎯 {len(st.session_state.portfolio_tickers)} tickers: {portfolio_display}")
    
    with col2:
        if st.button("💾 Save Portfolio"):
            portfolio_data = {
                'tickers': st.session_state.portfolio_tickers,
                'saved_at': datetime.now().isoformat()
            }
            with open('portfolio_config.json', 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            st.success("Portfolio saved!")
    
    st.markdown("---")
    
    # Add/Remove Tickers
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_ticker = st.text_input("Add Ticker", "", key="new_ticker").upper()
    
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Add") and new_ticker:
            if new_ticker not in st.session_state.portfolio_tickers:
                st.session_state.portfolio_tickers.append(new_ticker)
                save_portfolio_config()  # Save for model training sync
                st.success(f"Added {new_ticker}")
                st.rerun()
            else:
                st.warning(f"{new_ticker} already in portfolio")
    
    with col3:
        ticker_to_remove = st.selectbox("Remove", [""] + st.session_state.portfolio_tickers, key="remove_ticker")
        if st.button("➖ Remove") and ticker_to_remove:
            st.session_state.portfolio_tickers.remove(ticker_to_remove)
            save_portfolio_config()  # Save for model training sync
            st.success(f"Removed {ticker_to_remove}")
            st.rerun()
    
    st.markdown("---")
    
    # Signal Generation Section
    st.subheader("🚀 Generate Daily Signals")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Portfolio Size", len(st.session_state.portfolio_tickers))
    
    with col2:
        saved_strats, _ = load_saved_strategies()
        st.metric("Saved Strategies", len(saved_strats))
    
    with col3:
        last_run = st.session_state.get('last_signal_run', 'Never')
        if isinstance(last_run, str) and last_run != 'Never':
            last_run = datetime.fromisoformat(last_run).strftime('%m/%d %H:%M')
        st.metric("Last Run", last_run)
    
    # Per-Ticker Strategy Selection
    st.markdown("---")
    st.write("**📊 Strategy Selection per Ticker:**")
    
    if not saved_strats:
        st.warning("⚠️ No saved strategies found! Please optimize and save strategies first.")
    else:
        # Sort strategies by return (highest first)
        sorted_strats = sorted(saved_strats, key=lambda x: x['performance']['total_return_pct'], reverse=True)
        
        # Create strategy options for dropdown
        strategy_options = []
        for strat in sorted_strats:
            return_pct = strat['performance']['total_return_pct']
            ticker = strat.get('ticker', 'Unknown')
            period = strat.get('period', '1y')
            name_short = strat['name'][:40] + "..." if len(strat['name']) > 40 else strat['name']
            label = f"{return_pct:+.1f}% | {ticker} {period} | {name_short}"
            strategy_options.append((label, strat['timestamp']))
        
        # Initialize strategy selections in session state
        if 'ticker_strategy_map' not in st.session_state:
            st.session_state.ticker_strategy_map = {}
        
        # Quick selection buttons
        st.write("**Quick Actions:**")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("All → Highest Return", use_container_width=True):
                highest_timestamp = sorted_strats[0]['timestamp']
                for ticker in st.session_state.portfolio_tickers:
                    st.session_state.ticker_strategy_map[ticker] = highest_timestamp
                st.rerun()
        with col_btn2:
            if st.button("All → Most Recent", use_container_width=True):
                most_recent = max(saved_strats, key=lambda x: x['timestamp'])
                for ticker in st.session_state.portfolio_tickers:
                    st.session_state.ticker_strategy_map[ticker] = most_recent['timestamp']
                st.rerun()
        with col_btn3:
            if st.button("Reset All", use_container_width=True):
                st.session_state.ticker_strategy_map = {}
                st.rerun()
        
        st.markdown("---")
        
        # Display dropdown for each ticker
        st.info("💡 Select which strategy to use for each ticker. Strategies are sorted by return (highest first).")
        
        for ticker in st.session_state.portfolio_tickers:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{ticker}**")
            with col2:
                # Default to highest return strategy for THIS specific ticker
                default_idx = 0
                if ticker in st.session_state.ticker_strategy_map:
                    # Find index of previously selected strategy
                    selected_timestamp = st.session_state.ticker_strategy_map[ticker]
                    try:
                        default_idx = [s[1] for s in strategy_options].index(selected_timestamp)
                    except ValueError:
                        default_idx = 0
                else:
                    # Not set yet - find highest return strategy optimized on THIS ticker
                    ticker_strategies = [(i, strat) for i, strat in enumerate(sorted_strats) 
                                        if strat.get('ticker', '').upper() == ticker.upper()]
                    if ticker_strategies:
                        # Use the highest return strategy for this ticker
                        default_idx = ticker_strategies[0][0]
                    # else: default_idx stays 0 (overall highest if no ticker match)
                
                selected_label = st.selectbox(
                    f"Strategy for {ticker}",
                    options=[opt[0] for opt in strategy_options],
                    index=default_idx,
                    key=f"strategy_select_{ticker}",
                    label_visibility="collapsed"
                )
                
                # Store selection
                selected_idx = [opt[0] for opt in strategy_options].index(selected_label)
                st.session_state.ticker_strategy_map[ticker] = strategy_options[selected_idx][1]
    
    if st.button("🎯 Generate Signals for Portfolio", type="primary", use_container_width=True):
        st.session_state.trade_signals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tickers = len(st.session_state.portfolio_tickers)
        
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            status_text.text(f"Analyzing {ticker}... ({idx+1}/{total_tickers})")
            progress_bar.progress((idx + 1) / total_tickers)
            
            try:
                # Download data
                data = load_and_validate_data(ticker, period='1y', interval='1d')
                
                if data is None:
                    st.session_state.trade_signals.append({
                        'ticker': ticker,
                        'signal': 'ERROR',
                        'price': 0,
                        'error': 'No data available'
                    })
                    continue
                
                # Calculate indicators
                enriched_data = get_all_indicators(data)
                
                # Load strategy based on per-ticker selection
                if saved_strats:
                    # Get the selected strategy for this ticker
                    if ticker in st.session_state.ticker_strategy_map:
                        selected_timestamp = st.session_state.ticker_strategy_map[ticker]
                        # Find the strategy with matching timestamp
                        best_strategy = next((s for s in saved_strats if s['timestamp'] == selected_timestamp), None)
                        if best_strategy is None:
                            # Fallback to highest return if selected strategy not found
                            best_strategy = max(saved_strats, key=lambda x: x['performance']['total_return_pct'])
                    else:
                        # Default to highest return if no selection made
                        best_strategy = max(saved_strats, key=lambda x: x['performance']['total_return_pct'])
                    
                    # Generate signals
                    from optimization import universal_strategy
                    signals = universal_strategy(enriched_data, best_strategy['parameters'])
                    
                    # BACKTEST THE STRATEGY ON THIS TICKER - FULL YEAR
                    year_return = 0.0
                    year_trades = 0
                    try:
                        backtester_year = Backtester(
                            enriched_data,
                            best_strategy['name'],
                            universal_strategy,
                            best_strategy['parameters'],
                            100000  # Starting capital
                        )
                        backtester_year.run()
                        trade_log_year, summary_year = backtester_year.get_results()
                        
                        if len(trade_log_year) > 0:
                            year_return = summary_year['total_return_pct']
                            year_trades = len(trade_log_year)
                        else:
                            baseline_year = calculate_buy_and_hold_baseline(enriched_data, 100000)
                            year_return = baseline_year['total_return_pct']
                    except:
                        try:
                            baseline_year = calculate_buy_and_hold_baseline(enriched_data, 100000)
                            year_return = baseline_year['total_return_pct']
                        except:
                            year_return = 0.0
                    
                    # BACKTEST THE STRATEGY - LAST 30 DAYS (to match typical optimization period)
                    recent_return = 0.0
                    recent_trades = 0
                    return_note = ""
                    trade_log_recent = pd.DataFrame()
                    recent_data_for_chart = None
                    
                    try:
                        # Filter to last 30 days of data (matches optimization window)
                        current_date = enriched_data.index[-1]
                        days_ago_30 = current_date - pd.Timedelta(days=30)
                        recent_data = enriched_data[enriched_data.index >= days_ago_30]
                        recent_data_for_chart = recent_data.copy()  # Save for chart
                        
                        if len(recent_data) > 10:  # Need enough data for backtest
                            backtester_recent = Backtester(
                                recent_data,
                                best_strategy['name'],
                                universal_strategy,
                                best_strategy['parameters'],
                                100000
                            )
                            backtester_recent.run()
                            trade_log_recent, summary_recent = backtester_recent.get_results()
                            
                            if len(trade_log_recent) > 0:
                                recent_return = summary_recent['total_return_pct']
                                recent_trades = len(trade_log_recent)
                                return_note = f"{recent_trades} trades in last 30 days"
                            else:
                                baseline_recent = calculate_buy_and_hold_baseline(recent_data, 100000)
                                recent_return = baseline_recent['total_return_pct']
                                return_note = "Buy-and-hold (no trades in last 30 days)"
                        else:
                            recent_return = 0.0
                            return_note = "Insufficient data for recent period"
                            
                    except Exception as e:
                        try:
                            baseline_recent = calculate_buy_and_hold_baseline(recent_data, 100000)
                            recent_return = baseline_recent['total_return_pct']
                            return_note = "Buy-and-hold (last 30 days)"
                        except:
                            recent_return = 0.0
                            return_note = "Error calculating recent return"
                    
                    # Get most recent signal
                    last_signal = signals.iloc[-1]
                    last_close = enriched_data['close'].iloc[-1]
                    last_date = enriched_data.index[-1]
                    
                    # Price changes
                    price_change_1d = ((enriched_data['close'].iloc[-1] / enriched_data['close'].iloc[-2]) - 1) * 100 if len(enriched_data) >= 2 else 0
                    price_change_5d = ((enriched_data['close'].iloc[-1] / enriched_data['close'].iloc[-5]) - 1) * 100 if len(enriched_data) >= 5 else 0
                    
                    signal_type = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
                    
                    # Get original strategy performance (from when it was saved/optimized)
                    original_return = best_strategy['performance'].get('total_return_pct', 0)
                    original_ticker = best_strategy.get('ticker', 'Unknown')
                    original_period = best_strategy.get('period', '1y')
                    
                    signal_data = {
                        'ticker': ticker,
                        'signal': signal_type,
                        'price': round(last_close, 2),
                        'price_change_1d': round(price_change_1d, 2),
                        'price_change_5d': round(price_change_5d, 2),
                        'strategy': best_strategy['name'],
                        'original_return': round(original_return, 2),
                        'original_ticker': original_ticker,
                        'month_return': round(recent_return, 2),
                        'year_return': round(year_return, 2),
                        'original_label': f"Optimized ({original_period} on {original_ticker})",
                        'month_label': "Last 30 Days (Current)",
                        'year_label': "Full Year Backtest",
                        'return_note': return_note,
                        'date': last_date.strftime('%Y-%m-%d'),
                        'timestamp': datetime.now().isoformat(),
                        # Chart data (not serialized)
                        'trade_log_30d': trade_log_recent,
                        'price_data_30d': recent_data_for_chart
                    }
                    
                    st.session_state.trade_signals.append(signal_data)
                else:
                    st.session_state.trade_signals.append({
                        'ticker': ticker,
                        'signal': 'ERROR',
                        'price': 0,
                        'error': 'No strategies saved'
                    })
                    
            except Exception as e:
                st.session_state.trade_signals.append({
                    'ticker': ticker,
                    'signal': 'ERROR',
                    'price': 0,
                    'error': str(e)
                })
        
        st.session_state.last_signal_run = datetime.now().isoformat()
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Generated signals for {total_tickers} tickers!")
        st.rerun()
    
    # Display Signals
    if st.session_state.trade_signals:
        st.markdown("---")
        st.subheader("📊 Today's Signals")
        
        # Add explanation about returns
        st.info("""
        **💡 Understanding the THREE Performance Metrics:**
        
        **1. Optimized (1y on MSTY)** - Your original optimization result:
        - Shows the period you optimized on (e.g., 1y, 3mo, 1mo)
        - Shows which ticker (e.g., MSTY, SPY)
        - This is what convinced you to save this strategy!
        - Example: +115.8% on 30 days of MSTY data
        
        **2. Last 30 Days (Current)** - Recent performance on THIS ticker:
        - Tests strategy on LAST 30 DAYS of THIS ticker
        - Tells you if strategy works RIGHT NOW
        - ✅ Positive = Working now | ❌ Negative = Not working
        
        **3. Full Year Backtest** - Long-term test on THIS ticker:
        - Tests strategy on FULL 365 DAYS of THIS ticker
        - Shows long-term reliability
        - **Often differs from original if you optimized on shorter period!**
        
        **CRITICAL: Why Original and Full Year differ:**
        Example: You optimized on 30 days (Oct 12 - Nov 11):
        - Found patterns that worked great → +115.8% ✅
        - But those patterns only existed in that specific 30-day window!
        - Full year test (365 days) includes 11 other months where it failed → -23.5% ❌
        - **This is NORMAL!** Short-term optimizations often fail long-term.
        
        **What to look for:**
        - ✅ All three positive = Robust strategy (rare!)
        - ✅ Original & 30-day positive, 1-year negative = Works short-term (trade cautiously)
        - ❌ 30-day negative = Don't trade (even if original was great)
        """)
        
        # Filter signals
        buy_signals = [s for s in st.session_state.trade_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in st.session_state.trade_signals if s['signal'] == 'SELL']
        hold_signals = [s for s in st.session_state.trade_signals if s['signal'] == 'HOLD']
        error_signals = [s for s in st.session_state.trade_signals if s['signal'] == 'ERROR']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 BUY", len(buy_signals))
        with col2:
            st.metric("🔴 SELL", len(sell_signals))
        with col3:
            st.metric("⚪ HOLD", len(hold_signals))
        with col4:
            st.metric("❌ ERRORS", len(error_signals))
        
        # Display signals in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🟢 BUY Signals", "🔴 SELL Signals", "⚪ HOLD Signals", "📊 All Signals"])
        
        with tab1:
            if buy_signals:
                for signal in buy_signals:
                    with st.container():
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 1.5, 2, 2, 2])
                        with col1:
                            st.markdown(f"### {signal['ticker']}")
                        with col2:
                            st.metric("Price", f"${signal['price']:.2f}", f"{signal['price_change_1d']:+.2f}%")
                        with col3:
                            st.metric("5-Day", f"{signal['price_change_5d']:+.2f}%")
                        with col4:
                            original_label = signal.get('original_label', 'Strategy Return')
                            st.metric(original_label, f"{signal.get('original_return', 0):.1f}%")
                        with col5:
                            month_label = signal.get('month_label', 'Last 30 Days')
                            st.metric(month_label, f"{signal.get('month_return', 0):.1f}%")
                        with col6:
                            year_label = signal.get('year_label', '1-Year')
                            st.metric(year_label, f"{signal.get('year_return', 0):.1f}%")
                        st.caption(f"Strategy: {signal['strategy']}")
                        if 'return_note' in signal:
                            st.caption(f"📊 {signal['return_note']}")
                        
                        # Add 30-day performance chart
                        if 'price_data_30d' in signal and signal['price_data_30d'] is not None:
                            with st.expander("📈 View 30-Day Performance Chart", expanded=False):
                                chart = create_30day_performance_chart(
                                    signal['ticker'],
                                    signal['price_data_30d'],
                                    signal.get('trade_log_30d', pd.DataFrame()),
                                    signal['strategy'],
                                    signal['month_return']
                                )
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                        
                        st.markdown("---")
            else:
                st.info("No BUY signals today")
        
        with tab2:
            if sell_signals:
                for signal in sell_signals:
                    with st.container():
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 1.5, 2, 2, 2])
                        with col1:
                            st.markdown(f"### {signal['ticker']}")
                        with col2:
                            st.metric("Price", f"${signal['price']:.2f}", f"{signal['price_change_1d']:+.2f}%")
                        with col3:
                            st.metric("5-Day", f"{signal['price_change_5d']:+.2f}%")
                        with col4:
                            original_label = signal.get('original_label', 'Strategy Return')
                            st.metric(original_label, f"{signal.get('original_return', 0):.1f}%")
                        with col5:
                            month_label = signal.get('month_label', 'Last 30 Days')
                            st.metric(month_label, f"{signal.get('month_return', 0):.1f}%")
                        with col6:
                            year_label = signal.get('year_label', '1-Year')
                            st.metric(year_label, f"{signal.get('year_return', 0):.1f}%")
                        st.caption(f"Strategy: {signal['strategy']}")
                        if 'return_note' in signal:
                            st.caption(f"📊 {signal['return_note']}")
                        
                        # Add 30-day performance chart
                        if 'price_data_30d' in signal and signal['price_data_30d'] is not None:
                            with st.expander("📈 View 30-Day Performance Chart", expanded=False):
                                chart = create_30day_performance_chart(
                                    signal['ticker'],
                                    signal['price_data_30d'],
                                    signal.get('trade_log_30d', pd.DataFrame()),
                                    signal['strategy'],
                                    signal['month_return']
                                )
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                        
                        st.markdown("---")
            else:
                st.info("No SELL signals today")
        
        with tab3:
            if hold_signals:
                for signal in hold_signals:
                    with st.container():
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 1.5, 2, 2, 2])
                        with col1:
                            st.markdown(f"### {signal['ticker']}")
                        with col2:
                            st.metric("Price", f"${signal['price']:.2f}", f"{signal['price_change_1d']:+.2f}%")
                        with col3:
                            st.metric("5-Day", f"{signal['price_change_5d']:+.2f}%")
                        with col4:
                            original_label = signal.get('original_label', 'Strategy Return')
                            st.metric(original_label, f"{signal.get('original_return', 0):.1f}%")
                        with col5:
                            month_label = signal.get('month_label', 'Last 30 Days')
                            st.metric(month_label, f"{signal.get('month_return', 0):.1f}%")
                        with col6:
                            year_label = signal.get('year_label', '1-Year')
                            st.metric(year_label, f"{signal.get('year_return', 0):.1f}%")
                        st.caption(f"Strategy: {signal['strategy']}")
                        if 'return_note' in signal:
                            st.caption(f"📊 {signal['return_note']}")
                        
                        # Add 30-day performance chart
                        if 'price_data_30d' in signal and signal['price_data_30d'] is not None:
                            with st.expander("📈 View 30-Day Performance Chart", expanded=False):
                                chart = create_30day_performance_chart(
                                    signal['ticker'],
                                    signal['price_data_30d'],
                                    signal.get('trade_log_30d', pd.DataFrame()),
                                    signal['strategy'],
                                    signal['month_return']
                                )
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                        
                        st.markdown("---")
            else:
                st.info("No HOLD signals today")
        
        with tab4:
            # Create DataFrame - exclude non-serializable fields
            serializable_signals = []
            for signal in st.session_state.trade_signals:
                # Create a copy without the DataFrame fields
                signal_copy = {k: v for k, v in signal.items() 
                              if k not in ['trade_log_30d', 'price_data_30d']}
                serializable_signals.append(signal_copy)
            
            df = pd.DataFrame(serializable_signals)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"signals_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON Export - exclude DataFrames that aren't JSON serializable
                serializable_signals = []
                for signal in st.session_state.trade_signals:
                    # Create a copy without the DataFrame fields
                    signal_copy = {k: v for k, v in signal.items() 
                                  if k not in ['trade_log_30d', 'price_data_30d']}
                    serializable_signals.append(signal_copy)
                
                json_str = json.dumps(serializable_signals, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"signals_{datetime.now().strftime('%Y-%m-%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Alert Settings Section
    st.markdown("---")
    st.subheader("📱 Alert Settings")
    
    with st.expander("⚙️ Configure Alerts"):
        st.write("### SMS Alerts (via Twilio)")
        st.info("📱 Get instant text messages for BUY/SELL signals")
        
        sms_enabled = st.checkbox("Enable SMS Alerts", value=st.session_state.alert_settings['sms_enabled'])
        
        if sms_enabled:
            st.write("**Twilio Configuration:**")
            st.caption("Get free Twilio account at: https://www.twilio.com/try-twilio")
            
            col1, col2 = st.columns(2)
            with col1:
                account_sid = st.text_input("Twilio Account SID", value=st.session_state.alert_settings['twilio_account_sid'], type="password")
                auth_token = st.text_input("Twilio Auth Token", value=st.session_state.alert_settings['twilio_auth_token'], type="password")
            
            with col2:
                twilio_phone = st.text_input("Twilio Phone Number", value=st.session_state.alert_settings['twilio_phone_number'], placeholder="+1234567890")
                your_phone = st.text_input("Your Phone Number", value=st.session_state.alert_settings['phone_number'], placeholder="+1234567890")
            
            if st.button("💾 Save SMS Settings"):
                st.session_state.alert_settings['sms_enabled'] = sms_enabled
                st.session_state.alert_settings['twilio_account_sid'] = account_sid
                st.session_state.alert_settings['twilio_auth_token'] = auth_token
                st.session_state.alert_settings['twilio_phone_number'] = twilio_phone
                st.session_state.alert_settings['phone_number'] = your_phone
                st.success("SMS settings saved!")
            
            # Send test SMS
            if st.button("📱 Send Test SMS"):
                try:
                    from twilio.rest import Client
                    client = Client(account_sid, auth_token)
                    message = client.messages.create(
                        body=f"🎯 Pattern_FindR Test Alert\n\nThis is a test message. You're all set up!",
                        from_=twilio_phone,
                        to=your_phone
                    )
                    st.success(f"✅ Test SMS sent! Message SID: {message.sid}")
                except Exception as e:
                    st.error(f"❌ Error sending SMS: {e}")
                    st.caption("Make sure Twilio credentials are correct and phone numbers are in E.164 format (+1234567890)")
        
        st.markdown("---")
        st.write("### Email Alerts")
        st.info("📧 Get email notifications for signals")
        st.warning("⚠️ Email alerts coming soon! For now, use SMS or check the app daily.")
    
    st.stop()

# Display saved strategies if requested
if st.session_state.get('show_saved_strategies', False):
    st.write("# 💾 Saved Strategies")
    
    col_back, col_cache, col_delete = st.columns([1, 2, 2])
    with col_back:
        if st.button("⬅️ Back to Main"):
            st.session_state.show_saved_strategies = False
            st.rerun()
    with col_cache:
        if st.button("🔄 Clear Cache & Reload"):
            st.cache_data.clear()
            st.success("Cache cleared! Reloading strategies...")
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete ALL Strategies", type="secondary"):
            import shutil
            storage_dir = "saved_strategies"
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
                os.makedirs(storage_dir)
                st.cache_data.clear()
                st.success("✅ All strategies deleted! Starting fresh.")
                st.rerun()
            else:
                st.warning("No strategies to delete.")
    
    # Info box for old strategies
    st.info("""
    **💡 Seeing "Unknown" ticker?** Your strategies were saved before ticker tracking was added.
    
    **Fix options:**
    1. **Quick fix:** Run `python fix_old_strategies.py` to update all at once
    2. **Or:** Delete old strategies and re-optimize with current version
    3. **Or:** Manually edit JSON files in `saved_strategies/` folder
    """)
    
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
        # Group strategies by period
        from collections import defaultdict
        period_groups = defaultdict(list)
        for strategy in saved_strategies:
            period = strategy.get('period', 'Unknown')
            period_groups[period].append(strategy)
        
        # Define period order for display
        period_order = ['5y', '3y', '2y', '1y', '6mo', '3mo', '1mo', '5d', 'Unknown']
        sorted_periods = sorted(period_groups.keys(), 
                               key=lambda x: period_order.index(x) if x in period_order else len(period_order))
        
        # Display strategies grouped by period
        for period in sorted_periods:
            strategies_in_period = period_groups[period]
            
            # Period header with count
            period_icon = "📅" if period != 'Unknown' else "❓"
            st.markdown(f"### {period_icon} {period.upper()} Strategies ({len(strategies_in_period)})")
            
            for strategy in strategies_in_period:
                return_pct = strategy['performance']['total_return_pct']
                return_icon = "📈" if return_pct > 0 else "📉"
                
                # Display ticker in expander title
                ticker_info = f"📊 {strategy.get('ticker', 'Unknown')}"
                period_info = f"{strategy.get('period', '1y')} {strategy.get('interval', '1d')}"
                
                with st.expander(f"{return_icon} {ticker_info} - {strategy['name']} - {return_pct:.2f}% Return ({len(strategy['active_indicators'])} indicators)"):
                    # Add ticker/period banner
                    st.info(f"**Optimized on:** {strategy.get('ticker', 'Unknown')} | **Period:** {period_info}")
                    
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
                    
                    # Add visualization and trade details
                    st.markdown("---")
                
                    # Create tabs for details
                    detail_tabs = st.tabs(["📊 Performance Chart", "📋 Trade Log", "🔧 Parameters"])
                    
                    with detail_tabs[0]:
                        st.write("**Performance Visualization:**")
                    
                        # Check if this is an old strategy without ticker info
                        strat_ticker = strategy.get('ticker', 'Unknown')
                        if strat_ticker == 'Unknown':
                            st.warning("""
                        ⚠️ **Old Strategy - No Ticker Information**
                        
                        This strategy was saved before ticker tracking was added.
                        
                        **To fix this:**
                        1. Remember which ticker you optimized this on (e.g., MSTY)
                        2. Re-optimize and save a new version
                        3. Or manually edit the JSON file to add: `"ticker": "MSTY"`
                        
                        **For now, visualization is disabled.**
                            """)
                        else:
                            # Need to reload data and regenerate chart
                            try:
                                # Load the data for this strategy
                                strat_period = strategy.get('period', '1y')
                                strat_interval = strategy.get('interval', '1d')
                                
                                with st.spinner(f"Loading {strat_ticker} data..."):
                                    strat_data = load_and_validate_data(strat_ticker, strat_period, strat_interval)
                            
                                if strat_data is not None:
                                    # Get indicators
                                    strat_enriched = get_all_indicators(strat_data)
                                    
                                    # Re-run backtest with saved parameters
                                    from optimization import universal_strategy
                                    backtester = Backtester(
                                        strat_enriched, 
                                        strategy['name'],
                                        universal_strategy,
                                        strategy['parameters'],
                                        strategy['performance']['starting_capital']
                                    )
                                    backtester.run()
                                    trade_log, summary = backtester.get_results()
                                
                                    # Calculate baseline
                                    baseline = calculate_buy_and_hold_baseline(
                                        strat_enriched, 
                                        strategy['performance']['starting_capital']
                                    )
                                    
                                    # Create chart
                                    fig = create_strategy_chart(strat_enriched, trade_log, strategy['name'], summary, baseline)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Check for recent trade activity
                                    if len(trade_log) > 0:
                                        last_exit = pd.to_datetime(trade_log.iloc[-1]['exit_date'])
                                        days_since_last_trade = (datetime.now() - last_exit).days
                                        
                                        if days_since_last_trade > 30:
                                            st.warning(f"""
                                            ⚠️ **No Recent Trades Detected**
                                            
                                            Last trade exit: **{last_exit.strftime('%Y-%m-%d')}** ({days_since_last_trade} days ago)
                                            
                                            **Possible reasons:**
                                            - 📊 **Indicator thresholds not being met** in current market conditions
                                            - 🔒 **Filters blocking trades** (trend filter, confirmation required)
                                            - 📉 **Market regime changed** since optimization
                                            - 🎯 **Strategy is very selective** (designed for fewer trades)
                                            
                                            **What to do:**
                                            1. Check "Active Indicators" tab to see current values
                                            2. Try "Generate Signals" to see if there are pending signals today
                                            3. Consider re-optimizing with current market data
                                            4. Check if `use_trend_filter` is blocking (ADX < 25 or wrong Supertrend)
                                            
                                            💡 **Tip:** If strategy parameters include `min_hold_days: 7+` and `use_trend_filter: True`,
                                            it will be VERY selective and may not trade during choppy/ranging markets.
                                            """)
                                            
                                            # Show diagnostic info
                                            with st.expander("🔍 **Diagnostics: Why No Recent Trades?**"):
                                                st.write("**Strategy Parameters:**")
                                                params = strategy.get('parameters', {})
                                                
                                                # Check key filter parameters
                                                st.write(f"- Min Hold Days: **{params.get('min_hold_days', 'N/A')}** days")
                                                st.write(f"- Require Confirmation: **{params.get('require_confirmation', 'N/A')}**")
                                                st.write(f"- Use Trend Filter: **{params.get('use_trend_filter', 'N/A')}**")
                                                st.write(f"- Buy Score Threshold: **{params.get('buy_score_threshold', 'N/A')}**")
                                                st.write(f"- Sell Score Threshold: **{params.get('sell_score_threshold', 'N/A')}**")
                                                
                                                st.write("\n**Current Indicator Values (Latest):**")
                                                if len(strat_enriched) > 0:
                                                    latest_row = strat_enriched.iloc[-1]
                                                    
                                                    # Check trend indicators
                                                    adx_cols = [c for c in strat_enriched.columns if 'ADX' in c]
                                                    if adx_cols:
                                                        adx_val = latest_row[adx_cols[0]]
                                                        st.write(f"- ADX: **{adx_val:.2f}** {'✅ Strong trend (>25)' if adx_val > 25 else '❌ Weak trend (<25)'}")
                                                    
                                                    # Check Supertrend
                                                    st_cols = [c for c in strat_enriched.columns if 'SUPERTd' in c]
                                                    if st_cols:
                                                        st_val = latest_row[st_cols[0]]
                                                        st.write(f"- Supertrend: **{'🟢 Uptrend' if st_val == 1 else '🔴 Downtrend'}**")
                                                    
                                                    # Check RSI
                                                    rsi_cols = [c for c in strat_enriched.columns if 'RSI' in c]
                                                    if rsi_cols:
                                                        rsi_val = latest_row[rsi_cols[0]]
                                                        status = '🟢 Oversold' if rsi_val < 30 else '🔴 Overbought' if rsi_val > 70 else '⚪ Neutral'
                                                        st.write(f"- RSI: **{rsi_val:.2f}** {status}")
                                                    
                                                    st.write("\n**Why this matters:**")
                                                    if params.get('use_trend_filter'):
                                                        st.write("- ⚠️ Trend filter is ENABLED - strategy only trades during strong trends (ADX > 25)")
                                                        if adx_cols and latest_row[adx_cols[0]] < 25:
                                                            st.write("- 🚫 **ADX < 25: All trades are currently BLOCKED**")
                                                    
                                                    if params.get('require_confirmation'):
                                                        st.write("- ⚠️ Confirmation required - needs 2 consecutive days of same signal")
                                                    
                                                    if params.get('min_hold_days', 0) > 5:
                                                        st.write(f"- ⚠️ High min_hold_days ({params['min_hold_days']}) - very selective, fewer trades")
                                else:
                                    st.warning("Unable to load market data for visualization")
                            except Exception as e:
                                st.error(f"Error generating chart: {e}")
                
                    with detail_tabs[1]:
                        st.write("**Trade Details:**")
                        if 'trade_log' in strategy and strategy['trade_log']:
                            trade_df = pd.DataFrame(strategy['trade_log'])
                            st.dataframe(trade_df, use_container_width=True)
                        else:
                            # Try to regenerate trade log
                            try:
                                strat_ticker = strategy.get('ticker', 'SPY')
                                strat_period = strategy.get('period', '1y')
                                strat_interval = strategy.get('interval', '1d')
                                
                                strat_data = load_and_validate_data(strat_ticker, strat_period, strat_interval)
                                if strat_data is not None:
                                    strat_enriched = get_all_indicators(strat_data)
                                    from optimization import universal_strategy
                                    backtester = Backtester(
                                        strat_enriched,
                                        strategy['name'],
                                        universal_strategy,
                                        strategy['parameters'],
                                        strategy['performance']['starting_capital']
                                    )
                                    backtester.run()
                                    trade_log, _ = backtester.get_results()
                                    
                                    if len(trade_log) > 0:
                                        st.dataframe(trade_log, use_container_width=True)
                                    else:
                                        st.info("No trades generated with this strategy")
                                else:
                                    st.warning("Unable to load data for trade details")
                            except Exception as e:
                                st.error(f"Error loading trade details: {e}")
                    
                    with detail_tabs[2]:
                        st.write("**Strategy Parameters:**")
                        param_data = []
                        for key, value in strategy['parameters'].items():
                            param_data.append({
                                'Parameter': key,
                                'Value': str(value)
                            })
                        st.table(param_data)
                    
                    st.markdown("---")
                    
                    # Action buttons
                    button_col1, button_col2 = st.columns([1, 4])
                    with button_col1:
                        # Use filename for unique key since timestamps might be identical
                        unique_key = strategy['filename'].replace('.json', '').replace('strategy_', '')
                        if st.button(f"🗑️ Delete", key=f"delete_{unique_key}"):
                            os.remove(strategy['filepath'])
                            st.success("Strategy deleted!")
                            st.rerun()
    else:
        if search_term or min_return is not None or max_return is not None:
            st.info("🔍 No strategies match your search criteria. Try adjusting your filters.")
        else:
            st.info("📁 No saved strategies found. Save some strategies from the main page first!")
    
    st.stop()  # Don't show the rest of the app when viewing saved strategies

# --- User Inputs ---
st.sidebar.header("User Inputs")

# Ticker input with quick-select options
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    ticker = st.text_input("Stock Ticker", "SPY").upper()
with col2:
    st.write("")
    st.write("Quick:")
    if st.button("MSTY", key="btn_msty"):
        ticker = "MSTY"
    if st.button("MSTR", key="btn_mstr"):
        ticker = "MSTR"

interval = st.sidebar.selectbox(
    "Select Timeframe",
    ['1m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
    index=5,  # Default to '1d'
    help="Intraday data (1m-1h) is limited to recent periods. Use daily (1d+) for longer history."
)

# Period selection with constraints for intraday data
if interval in ['1m', '5m', '15m', '30m', '1h']:
    # Intraday intervals require shorter periods
    if interval == '1m':
        period_options = ['1d', '5d', '7d']
        default_period = '5d'
        st.sidebar.info("⏰ 1-minute data limited to last 7 days")
    elif interval in ['5m', '15m', '30m']:
        period_options = ['1d', '5d', '1mo', '2mo']
        default_period = '1mo'
        st.sidebar.info("⏰ Intraday data limited to last 60 days")
    else:  # 1h
        period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y']
        default_period = '3mo'
        st.sidebar.info("⏰ Hourly data limited to last 730 days")
    
    period = st.sidebar.selectbox(
        "Select Period",
        period_options,
        index=period_options.index(default_period)
    )
else:
    # Daily and higher intervals can use any period
    period = st.sidebar.selectbox(
        "Select Period",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
        index=3  # Default to '1y'
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

# --- Trade Section ---
st.sidebar.header("🎯 Trade")

if st.sidebar.button("📊 Portfolio & Daily Signals", use_container_width=True):
    st.session_state.show_trade_section = True
    st.rerun()

# Load portfolio info for sidebar display
portfolio_file = 'portfolio_config.json'
if os.path.exists(portfolio_file):
    try:
        with open(portfolio_file, 'r') as f:
            portfolio_data = json.load(f)
            st.sidebar.write(f"**Portfolio:** {len(portfolio_data.get('tickers', []))} tickers")
    except:
        pass

st.sidebar.markdown("---")

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

# Number of trials selection
n_trials = st.sidebar.selectbox(
    "🎯 Number of Trials",
    options=[100, 500, 1000, 2500, 5000, 10000, 25000],
    index=4,  # Default to 5000
    help="More trials = better strategies but takes longer. Start with 1000 for testing."
)

# Trade Preference Slider (NEW!)
st.sidebar.subheader("📊 Trade Frequency Control")
trade_preference = st.sidebar.slider(
    "How many trades do you want?",
    min_value=0.0,
    max_value=1.0,
    value=0.4,  # Default to Conservative (fewer trades)
    step=0.1,
    help="""
    **🎯 Controls how selective strategies are:**
    
    • **0.0-0.3 = Very Conservative** (5-15 trades/year)
      Catches only the best setups, big moves
    
    • **0.4-0.6 = Balanced** (15-30 trades/year)
      Good mix of selectivity and activity
    
    • **0.7-1.0 = Aggressive** (30-60 trades/year)
      More active trading, smaller moves
    
    **💡 Tip:** Start with 0.3-0.4 to avoid overtrading!
    """
)

# Show trade preference label
if trade_preference < 0.3:
    pref_label = "🐢 Very Conservative"
    pref_desc = "~5-15 trades/year (best for swing trading)"
elif trade_preference < 0.4:
    pref_label = "🎯 Conservative"
    pref_desc = "~10-20 trades/year (recommended)"
elif trade_preference < 0.6:
    pref_label = "⚖️ Balanced"
    pref_desc = "~20-30 trades/year"
elif trade_preference < 0.8:
    pref_label = "🔥 Aggressive"
    pref_desc = "~30-50 trades/year"
else:
    pref_label = "⚡ Very Aggressive"
    pref_desc = "~50-80 trades/year"

st.sidebar.caption(f"{pref_label}: {pref_desc}")

# Show estimated time
estimated_time = (n_trials / n_jobs) / 60
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

# Position Sizing Toggle
st.sidebar.subheader("🎯 Position Sizing Strategy")
enable_position_sizing = st.sidebar.checkbox(
    "🚀 Enable Enhanced Position Sizing",
    value=True,
    help="Enable dynamic position sizing (50-100% per trade) with volatility adjustments and risk management. When disabled, uses 100% of available capital per trade (standard mode)."
)

if enable_position_sizing:
    st.sidebar.success("🚀 Enhanced Mode: Dynamic position sizing (50-100%)")
    st.sidebar.caption("• Volatility-based adjustments")
    st.sidebar.caption("• Confidence-weighted positions") 
    st.sidebar.caption("• Risk management features")
else:
    st.sidebar.info("📊 Standard Mode: Full capital per trade (100%)")
    st.sidebar.caption("• All available capital deployed")
    st.sidebar.caption("• Original backtester behavior")

# Benchmark option in sidebar
run_benchmark = st.sidebar.checkbox(
    "🏁 Run Benchmark First", 
    help="Test methods to find fastest (adds 2-3 min but optimizes the full run)"
)

# Live Trading Simulation Mode
st.sidebar.subheader("🔴 Live Trading Mode")
live_trading_mode = st.sidebar.checkbox(
    "🚨 Live Trading Simulation",
    value=False,
    help="Test on ONLY the most recent 30 days to verify real-time readiness. All indicators must work on TODAY's candle."
)

if live_trading_mode:
    st.sidebar.warning("⚠️ Live mode: Using only last 30 days!")
    st.sidebar.info("This tests if your strategy can trade TODAY")

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
                # Console output to help user see progress  
                print(f"\n🔄 CALCULATING INDICATORS FOR {ticker}")
                print(f"   Data range: {len(data)} rows")
                print(f"   Adding candlestick patterns...")
                
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

                print(f"   Adding technical indicators and advanced patterns...")
                enriched_data = get_all_indicators(data)
                print(f"✅ INDICATORS COMPLETE: {len(enriched_data.columns)} total indicators")
                
            st.success("Indicator and pattern calculation complete!")
            
            # LIVE TRADING MODE: Filter to most recent 30 days only
            if live_trading_mode:
                original_length = len(enriched_data)
                # Keep only the last 30 trading days
                enriched_data = enriched_data.iloc[-30:]
                st.warning(f"🔴 **LIVE TRADING SIMULATION MODE**")
                st.info(f"📊 Using only the most recent 30 trading days ({original_length - 30} days excluded)")
                st.info(f"✅ This verifies your strategy can trade TODAY with current indicators")
            
            # Show data range - date might be in index or column
            if 'date' in enriched_data.columns:
                data_start = enriched_data['date'].min()
                data_end = enriched_data['date'].max()
            else:
                # Date is in the index
                data_start = enriched_data.index.min()
                data_end = enriched_data.index.max()
            
            # Show data range with live trading indicator
            range_label = "🔴 LIVE MODE Data Range" if live_trading_mode else "📅 Data Range"
            st.info(f"**{range_label}:** {data_start.strftime('%B %d, %Y')} to {data_end.strftime('%B %d, %Y')} ({len(enriched_data)} trading days)")

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

            # Create progress tracking UI elements
            st.write(f"### 🚀 Running Optimization")
            st.write(f"**Method:** {optimization_method} | **Workers:** {n_jobs} | **Trials:** {n_trials:,}")
            
            # Import time for ETA calculation
            import time
            start_time = time.time()
            
            # Progress tracking depends on worker count
            if n_jobs == 1:
                # Single-threaded: Use progress bar (works perfectly)
                progress_bar = st.progress(0)
                status_text = st.empty()
                eta_text = st.empty()
                
                def update_progress(current_trial, total_trials):
                    progress = current_trial / total_trials
                    progress_bar.progress(min(progress, 1.0))
                    
                    elapsed = time.time() - start_time
                    if current_trial > 0:
                        avg_time_per_trial = elapsed / current_trial
                        remaining_trials = total_trials - current_trial
                        eta_seconds = avg_time_per_trial * remaining_trials
                        eta_minutes = eta_seconds / 60
                        
                        status_text.text(f"📊 Progress: {current_trial:,} / {total_trials:,} trials ({progress*100:.1f}%)")
                        eta_text.text(f"⏱️ Estimated time remaining: {eta_minutes:.1f} minutes")
                    else:
                        status_text.text(f"📊 Starting optimization...")
                        eta_text.text(f"⏱️ Calculating ETA...")
                
                progress_callback = update_progress
            else:
                # Multi-threaded: Progress callbacks don't work with parallel processes
                # Show terminal instructions instead
                st.info(f"🔄 **Optimization running with {n_jobs} parallel workers**")
                st.write("📺 **Watch progress in your terminal** - Streamlit can't update UI from parallel workers")
                st.write("You'll see a `tqdm` progress bar like this:")
                st.code("Parallel Trials: 45%|████████▌         | 4,500/10,000 [08:15<09:45,  9.41it/s]")
                st.write("⏳ This page will update automatically when optimization completes...")
                progress_callback = None
            
            # Start optimization with appropriate mode
            print(f"\n{'='*60}")
            print("🚨 OPTIMIZATION STARTING FROM STREAMLIT APP")
            print(f"{'='*60}")
            print(f"🎯 Ticker: {ticker}")
            print(f"📊 Trials: {n_trials:,}")
            print(f"⚙️  Workers: {n_jobs}")
            print(f"📋 Method: {optimization_method}")
            print(f"📈 Data points: {len(enriched_data):,}")
            print(f"🔧 Watch console for real-time progress...")
            print(f"{'='*60}\n")
            
            with st.spinner("Running optimization..."):
                # TensorFlow already configured for CPU-only at startup
                print(f"🔧 Running optimization with CPU-only TensorFlow")
                top_trials = run_optimization(enriched_data, n_trials=n_trials, n_jobs=n_jobs, progress_callback=progress_callback, trade_preference=trade_preference, enable_position_sizing=enable_position_sizing, ticker=ticker)
            
            # Show completion
            elapsed_total = time.time() - start_time
            st.success(f"✅ Completed {n_trials:,} trials in {elapsed_total/60:.1f} minutes")
            st.write(f"🎯 Found {len(top_trials)} successful strategies")
            
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
    
    # Show data range for clarity - date might be in index or column
    if 'date' in enriched_data.columns:
        data_start = enriched_data['date'].min()
        data_end = enriched_data['date'].max()
    else:
        # Date is in the index
        data_start = enriched_data.index.min()
        data_end = enriched_data.index.max()
    st.info(f"📅 **Data Range:** {data_start.strftime('%B %d, %Y')} to {data_end.strftime('%B %d, %Y')} ({len(enriched_data)} days)")
    
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
                
                # Use the same backtester type that was used during optimization
                if enable_position_sizing and 'max_position_pct' in final_params:
                    # Use enhanced backtester with position sizing (same as optimization)
                    from enhanced_backtester import EnhancedBacktester
                    backtester = EnhancedBacktester(
                        data=enriched_data,
                        strategy_name=strategy_name,
                        strategy_func=universal_strategy,
                        params=final_params,
                        starting_capital=starting_capital,
                        enable_position_sizing=True,
                        max_position_pct=final_params.get('max_position_pct', 0.50)
                    )
                else:
                    # Use standard backtester (100% capital deployment)
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
                    'baseline': None,  # Will be updated below
                    'ticker': ticker,
                    'period': period,
                    'interval': interval
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
