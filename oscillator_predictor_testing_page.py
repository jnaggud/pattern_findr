"""
Oscillator Predictor Testing Page
=================================
Comprehensive strategy validation and testing suite.

Features:
1. Walk-Forward Analysis - Rolling window optimization
2. Monte Carlo Simulation - Randomized trade sequences
3. Sensitivity Analysis - Parameter perturbation testing
4. Cross-Asset Testing - Multi-ticker validation
5. Market Regime Testing - Bull/bear/sideways analysis
6. Paper Trading Mode - Real-time signal verification
7. Statistical Significance Testing - Sharpe ratio confidence intervals
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ============================================================================
# REPLICATED FUNCTIONS FROM OSCILLATOR PREDICTOR PAGE
# ============================================================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI and normalize to -1 to +1 range"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R and normalize"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return (wr + 50) / 50


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI and normalize"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return (cci / 200).clip(-1, 1)


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """Calculate Stochastic Oscillator and normalize"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return (stoch_k - 50) / 50, (stoch_d - 50) / 50


def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change and normalize"""
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return (roc / 10).clip(-1, 1)


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum and normalize"""
    mom = close - close.shift(period)
    mom_std = mom.rolling(50).std()
    return (mom / (mom_std + 1e-10)).clip(-1, 1)


def calculate_bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Bollinger Band position (-1 to +1)"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    position = (close - lower) / (upper - lower + 1e-10)
    return (position - 0.5) * 2


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index and normalize"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_sum = money_flow_positive.rolling(window=period).sum()
    negative_sum = money_flow_negative.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
    return (mfi - 50) / 50


def create_composite_oscillator(data: pd.DataFrame) -> pd.DataFrame:
    """Create a composite oscillator from multiple normalized indicators."""
    df = data.copy()
    df.columns = df.columns.str.lower()

    components = {}
    components['rsi_norm'] = calculate_rsi(df['close'], 14)
    components['rsi_norm_7'] = calculate_rsi(df['close'], 7)
    components['willr_norm'] = calculate_williams_r(df['high'], df['low'], df['close'], 14)
    components['cci_norm'] = calculate_cci(df['high'], df['low'], df['close'], 20)
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    components['stoch_k_norm'] = stoch_k
    components['stoch_d_norm'] = stoch_d
    components['roc_norm'] = calculate_roc(df['close'], 10)
    components['momentum_norm'] = calculate_momentum(df['close'], 10)
    components['bb_position'] = calculate_bb_position(df['close'], 20)

    if 'volume' in df.columns and df['volume'].sum() > 0:
        components['mfi_norm'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)

    for name, series in components.items():
        df[name] = series

    composite = pd.Series(0.0, index=df.index)
    for name in components.keys():
        if name in df.columns:
            composite += df[name].fillna(0)

    df['composite_oscillator'] = composite / len(components)
    df['composite_smooth'] = df['composite_oscillator'].rolling(window=3, center=True).mean()
    df['composite_smooth'] = df['composite_smooth'].bfill().ffill()

    return df


def run_velocity_backtest(df: pd.DataFrame, params: dict, starting_capital: float = 100000) -> dict:
    """
    Run velocity-based backtest with given parameters.
    Returns metrics dictionary.
    """
    # Extract parameters
    signal_type = params.get('signal_type', 'velocity_crossover_and_zone')
    vel_smoothing = params.get('vel_smoothing', 3)
    oversold_threshold = params.get('oversold_threshold', -0.3)
    overbought_threshold = params.get('overbought_threshold', 0.3)
    stop_loss_pct = params.get('stop_loss_pct', 5.0)
    take_profit_pct = params.get('take_profit_pct', 10.0)
    min_bars_between = params.get('min_bars_between', 5)
    extreme_zone_mult = params.get('extreme_zone_mult', 1.5)
    exit_on_opposite_signal = params.get('exit_on_opposite_signal', True)
    exit_on_midline_cross = params.get('exit_on_midline_cross', False)

    # Advanced filter parameters
    rsi_filter = params.get('rsi_filter', 'none')
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    use_macd_confirm = params.get('use_macd_confirm', False)
    use_bb_filter = params.get('use_bb_filter', False)
    require_accel = params.get('require_accel', False)

    test_df = df.copy()

    # Get oscillator column
    osc_col = 'composite_smooth' if 'composite_smooth' in test_df.columns else 'composite_oscillator'

    # Smooth oscillator
    if vel_smoothing > 1:
        test_df['osc_smooth'] = test_df[osc_col].rolling(window=vel_smoothing, center=False).mean().bfill()
    else:
        test_df['osc_smooth'] = test_df[osc_col]

    # Calculate velocity and acceleration
    test_df['velocity'] = test_df['osc_smooth'].diff().fillna(0)
    test_df['acceleration'] = test_df['velocity'].diff().fillna(0)

    # Detect velocity zero-crossings
    test_df['vel_cross_up'] = (test_df['velocity'] > 0) & (test_df['velocity'].shift(1) <= 0)
    test_df['vel_cross_down'] = (test_df['velocity'] < 0) & (test_df['velocity'].shift(1) >= 0)

    # Build entry conditions
    osc_smooth = test_df['osc_smooth']
    velocity = test_df['velocity']

    in_oversold = osc_smooth < oversold_threshold
    in_overbought = osc_smooth > overbought_threshold
    extreme_oversold = osc_smooth < (oversold_threshold * extreme_zone_mult)
    extreme_overbought = osc_smooth > (overbought_threshold * extreme_zone_mult)

    vel_std = velocity.rolling(10, min_periods=1).std().fillna(velocity.std())
    strong_momentum_up = velocity > vel_std * 1.5
    strong_momentum_down = velocity < -vel_std * 1.5

    # Calculate RSI if needed
    rsi_buy_filter = pd.Series(True, index=test_df.index)
    rsi_sell_filter = pd.Series(True, index=test_df.index)
    if rsi_filter != 'none':
        delta = test_df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        test_df['rsi'] = rsi.fillna(50)

        if rsi_filter == 'both':
            rsi_buy_filter = test_df['rsi'] < rsi_oversold
            rsi_sell_filter = test_df['rsi'] > rsi_overbought
        elif rsi_filter == 'oversold_only':
            rsi_buy_filter = test_df['rsi'] < rsi_oversold
        elif rsi_filter == 'overbought_only':
            rsi_sell_filter = test_df['rsi'] > rsi_overbought

    # Calculate MACD if needed
    macd_buy_filter = pd.Series(True, index=test_df.index)
    if use_macd_confirm:
        ema12 = test_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = test_df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_buy_filter = macd_line > signal_line  # MACD above signal = bullish

    # Calculate Bollinger Bands if needed
    bb_buy_filter = pd.Series(True, index=test_df.index)
    if use_bb_filter:
        bb_sma = test_df['close'].rolling(window=20).mean()
        bb_std = test_df['close'].rolling(window=20).std()
        bb_lower = bb_sma - 2 * bb_std
        bb_buy_filter = test_df['close'] <= bb_lower  # Price at or below lower band

    # Acceleration filter
    accel_filter = pd.Series(True, index=test_df.index)
    if require_accel:
        accel_filter = test_df['acceleration'] > 0  # Positive acceleration for buys

    if signal_type == 'velocity_crossover_and_zone':
        buy_condition = test_df['vel_cross_up'] & in_oversold
        sell_condition = test_df['vel_cross_down'] & in_overbought
    elif signal_type == 'velocity_crossover_or_zone':
        buy_condition = test_df['vel_cross_up'] | extreme_oversold
        sell_condition = test_df['vel_cross_down'] | extreme_overbought
    elif signal_type == 'zone_only':
        buy_condition = extreme_oversold & (velocity > 0)
        sell_condition = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        buy_condition = strong_momentum_up & (osc_smooth < 0)
        sell_condition = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        buy_condition = test_df['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        sell_condition = test_df['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)
    else:
        buy_condition = test_df['vel_cross_up'] & in_oversold
        sell_condition = test_df['vel_cross_down'] & in_overbought

    # Apply all filters to buy/sell conditions
    buy_condition = buy_condition & rsi_buy_filter & macd_buy_filter & bb_buy_filter & accel_filter
    sell_condition = sell_condition & rsi_sell_filter

    test_df['buy_signal'] = buy_condition
    test_df['sell_signal'] = sell_condition

    # Run backtest
    position = 0
    entry_price = None
    entry_date = None
    last_trade_bar = -min_bars_between
    trades = []

    for i in range(len(test_df)):
        if i - last_trade_bar < min_bars_between:
            continue

        price = test_df['close'].iloc[i]
        date = test_df.index[i]

        # Check stop loss / take profit
        if position == 1 and entry_price is not None:
            pnl_pct = (price - entry_price) / entry_price * 100

            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'pnl': pnl_pct,
                    'exit_reason': 'stop_loss'
                })
                position = 0
                entry_price = None
                last_trade_bar = i
                continue

            if take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'pnl': pnl_pct,
                    'exit_reason': 'take_profit'
                })
                position = 0
                entry_price = None
                last_trade_bar = i
                continue

        # Check exit on opposite signal
        if position == 1 and exit_on_opposite_signal and test_df['sell_signal'].iloc[i]:
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'pnl': pnl_pct,
                'exit_reason': 'opposite_signal'
            })
            position = 0
            entry_price = None
            last_trade_bar = i
            continue

        # Check exit on midline cross (oscillator crosses above 0 for longs)
        if position == 1 and exit_on_midline_cross:
            osc_val = test_df['osc_smooth'].iloc[i]
            osc_prev = test_df['osc_smooth'].iloc[i-1] if i > 0 else osc_val
            if osc_prev <= 0 and osc_val > 0:  # Crossed above midline
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'pnl': pnl_pct,
                    'exit_reason': 'midline_cross'
                })
                position = 0
                entry_price = None
                last_trade_bar = i
                continue

        # Check entry
        if position == 0 and test_df['buy_signal'].iloc[i]:
            position = 1
            entry_price = price
            entry_date = date
            last_trade_bar = i

    # Close any open position
    if position == 1 and entry_price is not None:
        final_price = test_df['close'].iloc[-1]
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': test_df.index[-1],
            'exit_price': final_price,
            'pnl': pnl_pct,
            'exit_reason': 'end_of_period'
        })

    # Calculate metrics
    if not trades:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trades': [],
            'equity_curve': [starting_capital]
        }

    # Calculate equity curve
    equity = starting_capital
    equity_curve = [equity]
    peak_equity = equity
    max_drawdown = 0

    for trade in trades:
        equity = equity * (1 + trade['pnl'] / 100)
        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity - starting_capital) / starting_capital * 100
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_trade = np.mean(pnls) if pnls else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate Sharpe ratio (annualized)
    if len(pnls) > 1:
        returns_std = np.std(pnls)
        sharpe_ratio = (np.mean(pnls) / returns_std) * np.sqrt(252 / max(1, len(test_df) / len(trades))) if returns_std > 0 else 0
    else:
        sharpe_ratio = 0

    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades,
        'equity_curve': equity_curve,
        'ending_capital': equity
    }


@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Load data from yfinance with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None


# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

def run_walk_forward_analysis(df: pd.DataFrame, params: dict, n_splits: int = 5,
                              train_pct: float = 0.7, starting_capital: float = 100000) -> dict:
    """
    Run walk-forward analysis with rolling windows.

    Args:
        df: DataFrame with OHLCV data and composite oscillator
        params: Strategy parameters
        n_splits: Number of walk-forward splits
        train_pct: Percentage of each window for training
        starting_capital: Starting capital for each period

    Returns:
        Dictionary with walk-forward results
    """
    n_bars = len(df)
    window_size = n_bars // n_splits

    results = []

    for i in range(n_splits):
        start_idx = i * window_size
        end_idx = min((i + 2) * window_size, n_bars)  # Overlap windows

        if end_idx - start_idx < 50:  # Skip if too few bars
            continue

        window_df = df.iloc[start_idx:end_idx].copy()
        train_size = int(len(window_df) * train_pct)

        train_df = window_df.iloc[:train_size]
        test_df = window_df.iloc[train_size:]

        if len(test_df) < 10:
            continue

        # Run backtest on test portion only
        test_result = run_velocity_backtest(test_df, params, starting_capital)

        results.append({
            'split': i + 1,
            'train_start': train_df.index[0],
            'train_end': train_df.index[-1],
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1],
            'train_bars': len(train_df),
            'test_bars': len(test_df),
            'total_return': test_result['total_return'],
            'num_trades': test_result['num_trades'],
            'win_rate': test_result['win_rate'],
            'profit_factor': test_result['profit_factor'],
            'max_drawdown': test_result['max_drawdown'],
            'sharpe_ratio': test_result['sharpe_ratio'],
            'trades': test_result['trades']
        })

    # Aggregate results
    if not results:
        return {'splits': [], 'aggregate': {}}

    avg_return = np.mean([r['total_return'] for r in results])
    std_return = np.std([r['total_return'] for r in results])
    total_trades = sum([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
    avg_profit_factor = np.mean([r['profit_factor'] for r in results if r['profit_factor'] < float('inf')])
    avg_max_dd = np.mean([r['max_drawdown'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])

    # Calculate consistency score (% of profitable periods)
    profitable_splits = sum([1 for r in results if r['total_return'] > 0])
    consistency = profitable_splits / len(results) * 100

    return {
        'splits': results,
        'aggregate': {
            'avg_return': avg_return,
            'std_return': std_return,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'avg_max_drawdown': avg_max_dd,
            'avg_sharpe': avg_sharpe,
            'consistency': consistency,
            'n_splits': len(results)
        }
    }


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo_simulation(trades: list, n_simulations: int = 1000,
                               starting_capital: float = 100000) -> dict:
    """
    Run Monte Carlo simulation by randomizing trade order.

    Args:
        trades: List of trade dictionaries with 'pnl' key
        n_simulations: Number of simulations to run
        starting_capital: Starting capital

    Returns:
        Dictionary with simulation results
    """
    if not trades:
        return {'simulations': [], 'statistics': {}}

    pnls = [t['pnl'] for t in trades]
    n_trades = len(pnls)

    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Randomly shuffle trade order
        shuffled_pnls = np.random.permutation(pnls)

        # Calculate equity curve
        equity = starting_capital
        peak_equity = equity
        max_dd = 0

        for pnl in shuffled_pnls:
            equity = equity * (1 + pnl / 100)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd:
                max_dd = dd

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    # Calculate statistics
    returns = (final_equities - starting_capital) / starting_capital * 100

    return {
        'final_equities': final_equities,
        'max_drawdowns': max_drawdowns,
        'returns': returns,
        'statistics': {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_25': np.percentile(returns, 25),
            'percentile_75': np.percentile(returns, 75),
            'percentile_95': np.percentile(returns, 95),
            'prob_profit': np.sum(returns > 0) / len(returns) * 100,
            'mean_max_dd': np.mean(max_drawdowns),
            'worst_max_dd': np.max(max_drawdowns),
            'percentile_95_dd': np.percentile(max_drawdowns, 95)
        }
    }


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(df: pd.DataFrame, base_params: dict,
                            param_ranges: dict = None, n_steps: int = 5,
                            starting_capital: float = 100000) -> dict:
    """
    Run sensitivity analysis by perturbing parameters.

    Args:
        df: DataFrame with data
        base_params: Base strategy parameters
        param_ranges: Dict of {param_name: (min_mult, max_mult)} for perturbation
        n_steps: Number of steps for each parameter
        starting_capital: Starting capital

    Returns:
        Dictionary with sensitivity results
    """
    if param_ranges is None:
        # Default ranges for all numeric parameters
        param_ranges = {
            # Core thresholds
            'oversold_threshold': (0.7, 1.3),
            'overbought_threshold': (0.7, 1.3),
            # Risk management
            'stop_loss_pct': (0.5, 1.5),
            'take_profit_pct': (0.5, 1.5),
            # Timing parameters
            'vel_smoothing': (0.5, 2.0),
            'min_bars_between': (0.5, 2.0),
            'extreme_zone_mult': (0.7, 1.3),
            # RSI parameters (if used)
            'rsi_period': (0.5, 1.5),
            'rsi_oversold': (0.7, 1.3),
            'rsi_overbought': (0.7, 1.3),
        }

    results = {}
    base_result = run_velocity_backtest(df, base_params, starting_capital)
    base_return = base_result['total_return']

    for param_name, (min_mult, max_mult) in param_ranges.items():
        if param_name not in base_params:
            continue

        base_value = base_params[param_name]
        if base_value == 0:
            continue

        multipliers = np.linspace(min_mult, max_mult, n_steps)
        param_results = []

        for mult in multipliers:
            test_params = base_params.copy()
            new_value = base_value * mult

            # Handle integer parameters
            if param_name in ['vel_smoothing', 'min_bars_between', 'rsi_period', 'rsi_oversold', 'rsi_overbought']:
                new_value = max(1, int(round(new_value)))

            test_params[param_name] = new_value
            result = run_velocity_backtest(df, test_params, starting_capital)

            param_results.append({
                'multiplier': mult,
                'value': new_value,
                'total_return': result['total_return'],
                'num_trades': result['num_trades'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'return_change': result['total_return'] - base_return
            })

        # Calculate sensitivity score (std of returns / range of multipliers)
        returns = [r['total_return'] for r in param_results]
        sensitivity_score = np.std(returns) / (max_mult - min_mult) if len(returns) > 1 else 0

        results[param_name] = {
            'base_value': base_value,
            'results': param_results,
            'sensitivity_score': sensitivity_score,
            'max_return': max(returns),
            'min_return': min(returns),
            'return_range': max(returns) - min(returns)
        }

    # Rank parameters by sensitivity
    ranked_params = sorted(results.items(), key=lambda x: x[1]['sensitivity_score'], reverse=True)

    return {
        'base_return': base_return,
        'parameters': results,
        'ranked_sensitivity': [(p[0], p[1]['sensitivity_score']) for p in ranked_params]
    }


# ============================================================================
# CROSS-ASSET TESTING
# ============================================================================

def run_cross_asset_testing(tickers: list, params: dict, period: str = "2y",
                           starting_capital: float = 100000) -> dict:
    """
    Test strategy across multiple assets.

    Args:
        tickers: List of ticker symbols
        params: Strategy parameters
        period: Data period
        starting_capital: Starting capital

    Returns:
        Dictionary with cross-asset results
    """
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Testing {ticker}...")
        progress_bar.progress((i + 1) / len(tickers))

        try:
            df = load_data(ticker, period)
            if df is None or len(df) < 50:
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': 'Insufficient data'
                })
                continue

            # Create composite oscillator
            df = create_composite_oscillator(df)

            # Run backtest
            result = run_velocity_backtest(df, params, starting_capital)

            results.append({
                'ticker': ticker,
                'status': 'success',
                'total_return': result['total_return'],
                'num_trades': result['num_trades'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio'],
                'ending_capital': result['ending_capital']
            })
        except Exception as e:
            results.append({
                'ticker': ticker,
                'status': 'failed',
                'error': str(e)
            })

    progress_bar.empty()
    status_text.empty()

    # Aggregate successful results
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        return {'results': results, 'aggregate': {}}

    aggregate = {
        'n_assets': len(successful),
        'n_failed': len(results) - len(successful),
        'avg_return': np.mean([r['total_return'] for r in successful]),
        'std_return': np.std([r['total_return'] for r in successful]),
        'min_return': min([r['total_return'] for r in successful]),
        'max_return': max([r['total_return'] for r in successful]),
        'avg_win_rate': np.mean([r['win_rate'] for r in successful]),
        'avg_profit_factor': np.mean([r['profit_factor'] for r in successful if r['profit_factor'] < float('inf')]),
        'profitable_assets': sum([1 for r in successful if r['total_return'] > 0]),
        'pct_profitable': sum([1 for r in successful if r['total_return'] > 0]) / len(successful) * 100
    }

    return {'results': results, 'aggregate': aggregate}


# ============================================================================
# MARKET REGIME TESTING
# ============================================================================

def detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Detect market regime (bull, bear, sideways) based on trend and volatility.
    """
    close = df['close']

    # Calculate trend (SMA slope)
    sma = close.rolling(lookback).mean()
    sma_slope = (sma - sma.shift(lookback)) / sma.shift(lookback) * 100

    # Calculate volatility
    returns = close.pct_change()
    volatility = returns.rolling(lookback).std() * np.sqrt(252) * 100

    # Classify regimes
    regime = pd.Series('sideways', index=df.index)
    regime[sma_slope > 5] = 'bull'
    regime[sma_slope < -5] = 'bear'
    regime[(volatility > volatility.rolling(100).quantile(0.8)) & (abs(sma_slope) < 5)] = 'high_volatility'

    return regime


def run_regime_analysis(df: pd.DataFrame, params: dict, starting_capital: float = 100000) -> dict:
    """
    Analyze strategy performance across different market regimes.
    """
    # Detect regimes
    df = df.copy()
    df['regime'] = detect_market_regime(df)

    # Run full backtest first
    full_result = run_velocity_backtest(df, params, starting_capital)

    # Analyze by regime
    regime_results = {}

    for regime in ['bull', 'bear', 'sideways', 'high_volatility']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) < 30:
            continue

        # Find trades that occurred in this regime
        regime_trades = []
        for trade in full_result['trades']:
            entry_date = trade['entry_date']
            if entry_date in df.index and df.loc[entry_date, 'regime'] == regime:
                regime_trades.append(trade)

        if not regime_trades:
            regime_results[regime] = {
                'n_bars': len(regime_df),
                'n_trades': 0,
                'total_return': 0,
                'win_rate': 0
            }
            continue

        pnls = [t['pnl'] for t in regime_trades]
        wins = [p for p in pnls if p > 0]

        regime_results[regime] = {
            'n_bars': len(regime_df),
            'n_trades': len(regime_trades),
            'total_return': sum(pnls),
            'avg_trade': np.mean(pnls),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }

    return {
        'full_result': full_result,
        'regime_results': regime_results,
        'regime_distribution': df['regime'].value_counts().to_dict()
    }


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def calculate_statistical_significance(trades: list, n_bootstrap: int = 1000) -> dict:
    """
    Calculate statistical significance of strategy returns using bootstrap.
    """
    if len(trades) < 5:
        return {'significant': False, 'reason': 'Too few trades'}

    pnls = np.array([t['pnl'] for t in trades])
    n_trades = len(pnls)

    # Original statistics
    orig_mean = np.mean(pnls)
    orig_sharpe = orig_mean / np.std(pnls) if np.std(pnls) > 0 else 0

    # Bootstrap
    bootstrap_means = []
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(pnls, size=n_trades, replace=True)
        bootstrap_means.append(np.mean(sample))
        std = np.std(sample)
        bootstrap_sharpes.append(np.mean(sample) / std if std > 0 else 0)

    bootstrap_means = np.array(bootstrap_means)
    bootstrap_sharpes = np.array(bootstrap_sharpes)

    # Calculate confidence intervals
    mean_ci_lower = np.percentile(bootstrap_means, 2.5)
    mean_ci_upper = np.percentile(bootstrap_means, 97.5)
    sharpe_ci_lower = np.percentile(bootstrap_sharpes, 2.5)
    sharpe_ci_upper = np.percentile(bootstrap_sharpes, 97.5)

    # Test if significantly different from zero
    mean_significant = mean_ci_lower > 0 or mean_ci_upper < 0
    sharpe_significant = sharpe_ci_lower > 0 or sharpe_ci_upper < 0

    # T-test against zero
    t_stat, p_value = stats.ttest_1samp(pnls, 0)

    return {
        'n_trades': n_trades,
        'mean_return': orig_mean,
        'sharpe_ratio': orig_sharpe,
        'mean_ci_95': (mean_ci_lower, mean_ci_upper),
        'sharpe_ci_95': (sharpe_ci_lower, sharpe_ci_upper),
        'mean_significant': mean_significant,
        'sharpe_significant': sharpe_significant,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01
    }


# ============================================================================
# PAPER TRADING MODE
# ============================================================================

def get_paper_trading_signals(df: pd.DataFrame, params: dict) -> dict:
    """
    Generate current trading signals for paper trading.
    """
    # Get the last few bars
    recent_df = df.tail(20).copy()

    osc_col = 'composite_smooth' if 'composite_smooth' in recent_df.columns else 'composite_oscillator'
    vel_smoothing = params.get('vel_smoothing', 3)

    if vel_smoothing > 1:
        recent_df['osc_smooth'] = recent_df[osc_col].rolling(window=vel_smoothing, center=False).mean().bfill()
    else:
        recent_df['osc_smooth'] = recent_df[osc_col]

    recent_df['velocity'] = recent_df['osc_smooth'].diff().fillna(0)
    recent_df['acceleration'] = recent_df['velocity'].diff().fillna(0)

    # Get latest values
    latest = recent_df.iloc[-1]
    prev = recent_df.iloc[-2] if len(recent_df) > 1 else latest

    current_osc = latest['osc_smooth']
    current_velocity = latest['velocity']
    current_accel = latest['acceleration']
    current_price = latest['close']

    # Determine signal
    oversold_threshold = params.get('oversold_threshold', -0.3)
    overbought_threshold = params.get('overbought_threshold', 0.3)

    vel_cross_up = current_velocity > 0 and prev['velocity'] <= 0
    vel_cross_down = current_velocity < 0 and prev['velocity'] >= 0

    in_oversold = current_osc < oversold_threshold
    in_overbought = current_osc > overbought_threshold

    signal = 'HOLD'
    signal_strength = 'weak'

    if vel_cross_up and in_oversold:
        signal = 'BUY'
        signal_strength = 'strong'
    elif vel_cross_up:
        signal = 'BUY'
        signal_strength = 'moderate'
    elif vel_cross_down and in_overbought:
        signal = 'SELL'
        signal_strength = 'strong'
    elif vel_cross_down:
        signal = 'SELL'
        signal_strength = 'moderate'
    elif in_oversold and current_velocity > 0:
        signal = 'BUY'
        signal_strength = 'weak'
    elif in_overbought and current_velocity < 0:
        signal = 'SELL'
        signal_strength = 'weak'

    return {
        'timestamp': df.index[-1],
        'price': current_price,
        'oscillator': current_osc,
        'velocity': current_velocity,
        'acceleration': current_accel,
        'signal': signal,
        'signal_strength': signal_strength,
        'in_oversold': in_oversold,
        'in_overbought': in_overbought,
        'vel_cross_up': vel_cross_up,
        'vel_cross_down': vel_cross_down
    }


# ============================================================================
# CHART HELPER (replicated from main page)
# ============================================================================

class ChartHelper:
    """Helper class for creating gap-free financial charts."""

    def __init__(self, df: pd.DataFrame, interval: str):
        self.df = df
        self.interval = interval
        self.is_intraday = interval != "1d"

        if self.is_intraday:
            self.date_to_idx = {dt: i for i, dt in enumerate(df.index)}
        else:
            self.date_to_idx = None

    def get_x(self, dates):
        if not self.is_intraday:
            return dates

        if hasattr(dates, '__iter__') and not isinstance(dates, str):
            if hasattr(dates, 'tolist'):
                return [self.date_to_idx.get(d, None) for d in dates]
            else:
                return [self._lookup_date(d) for d in dates]
        else:
            return self._lookup_date(dates)

    def _lookup_date(self, dt):
        if dt in self.date_to_idx:
            return self.date_to_idx[dt]
        for ref_dt, idx in self.date_to_idx.items():
            if abs((ref_dt - dt).total_seconds()) < 3600:
                return idx
        return None

    def apply_formatting(self, fig):
        if not self.is_intraday:
            return

        n_ticks = min(12, len(self.df))
        step = max(1, len(self.df) // n_ticks)
        tickvals = list(range(0, len(self.df), step))
        ticktext = []
        for i in tickvals:
            if i < len(self.df):
                ticktext.append(self.df.index[i].strftime('%b %d\n%H:%M'))
        tickvals = tickvals[:len(ticktext)]

        fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)


# ============================================================================
# STREAMLIT PAGE
# ============================================================================

def render_oscillator_predictor_testing_page():
    """Main Streamlit page for strategy testing."""

    st.title("Oscillator Predictor Testing Suite")
    st.markdown("""
    **Comprehensive strategy validation and testing tools:**
    - Walk-Forward Analysis
    - Monte Carlo Simulation
    - Sensitivity Analysis
    - Cross-Asset Testing
    - Market Regime Analysis
    - Statistical Significance Testing
    - Paper Trading Mode
    """)

    # ========== SIDEBAR ==========
    st.sidebar.header("🧪 Testing Settings")

    # ========== LOAD SAVED STRATEGY (FIRST!) ==========
    st.sidebar.markdown("## 📂 Step 1: Load Strategy")
    st.sidebar.markdown("**Select a saved strategy to test:**")

    # Collect all available strategies from multiple locations
    all_strategies = {}

    # 1. Production config (currently deployed)
    if os.path.exists("production_env/velocity_config.json"):
        all_strategies["🚀 PRODUCTION (Currently Deployed)"] = "production_env/velocity_config.json"

    # 2. Velocity strategies directory - standalone JSON files
    if os.path.exists("velocity_strategies"):
        for item in os.listdir("velocity_strategies"):
            item_path = os.path.join("velocity_strategies", item)
            if item.endswith('.json') and os.path.isfile(item_path):
                name = item.replace('.json', '')
                all_strategies[f"📊 {name}"] = item_path

    # 3. Velocity strategies directory - bundle folders
    if os.path.exists("velocity_strategies"):
        for folder in os.listdir("velocity_strategies"):
            folder_path = os.path.join("velocity_strategies", folder)
            if os.path.isdir(folder_path):
                config_path = os.path.join(folder_path, "velocity_config.json")
                if os.path.exists(config_path):
                    all_strategies[f"📦 {folder}"] = config_path

    # 4. Old strategies directory
    if os.path.exists("strategies"):
        for folder in os.listdir("strategies"):
            folder_path = os.path.join("strategies", folder)
            if os.path.isdir(folder_path):
                config_path = os.path.join(folder_path, "velocity_config.json")
                if os.path.exists(config_path):
                    all_strategies[f"📁 {folder}"] = config_path

    # Initialize session state for loaded strategy
    if 'loaded_strategy_params' not in st.session_state:
        st.session_state['loaded_strategy_params'] = None
        st.session_state['loaded_strategy_name'] = None

    # Show count of available strategies
    num_strategies = len(all_strategies)
    if num_strategies == 0:
        st.sidebar.warning("⚠️ No saved strategies found! Save a strategy from the Oscillator Predictor page first.")
        strategy_options = ["(No strategies available)"]
    else:
        st.sidebar.success(f"✅ Found {num_strategies} saved strategies")
        strategy_options = ["-- Select a Strategy --"] + list(all_strategies.keys())

    selected_strategy = st.sidebar.selectbox(
        "🎯 Strategy to Test",
        strategy_options,
        index=0,
        key="strategy_selector",
        help="Select a strategy from Oscillator Predictor page to test with advanced validation"
    )

    loaded_params = None
    loaded_ticker = "SPY"

    if selected_strategy not in ["(No strategies available)", "-- Select a Strategy --", "(Manual Parameters)"]:
        config_path = all_strategies[selected_strategy]
        try:
            with open(config_path, 'r') as f:
                loaded_params = json.load(f)
            st.session_state['loaded_strategy_params'] = loaded_params
            st.session_state['loaded_strategy_name'] = selected_strategy

            # Extract ticker from loaded strategy
            loaded_ticker = loaded_params.get('ticker', 'SPY')

            # Display loaded strategy info
            st.sidebar.success(f"✅ Loaded: {loaded_params.get('strategy_name', 'Unknown')}")

            with st.sidebar.expander("📋 Strategy Details", expanded=True):
                st.write(f"**Ticker:** {loaded_params.get('ticker', 'N/A')}")
                st.write(f"**Interval:** {loaded_params.get('interval', '1d')}")
                # Show optimization period if available
                opt_period = loaded_params.get('optimization_period')
                opt_bars = loaded_params.get('optimization_bars')
                if opt_period:
                    st.write(f"**Optimized On:** {opt_period}" + (f" ({opt_bars} bars)" if opt_bars else ""))
                else:
                    st.write("**Optimized On:** Unknown (older strategy)")
                st.markdown("---")
                st.write(f"**Signal Type:** {loaded_params.get('signal_type', 'N/A')}")
                st.write(f"**Oversold:** {loaded_params.get('oversold_threshold', 'N/A'):.3f}")
                st.write(f"**Overbought:** {loaded_params.get('overbought_threshold', 'N/A'):.3f}")
                st.write(f"**Stop Loss:** {loaded_params.get('stop_loss_pct', 'N/A'):.1f}%")
                st.write(f"**Take Profit:** {loaded_params.get('take_profit_pct', 'N/A'):.1f}%")
                st.write(f"**Vel Smoothing:** {loaded_params.get('vel_smoothing', 'N/A')}")
                st.write(f"**Min Bars:** {loaded_params.get('min_bars_between', 'N/A')}")
                if loaded_params.get('saved_at'):
                    st.write(f"**Saved:** {loaded_params.get('saved_at', 'N/A')}")

        except Exception as e:
            st.sidebar.error(f"Error loading strategy: {e}")
            loaded_params = None
    else:
        # No strategy selected - show a prompt
        if num_strategies > 0:
            st.sidebar.info("👆 Select a strategy above to begin testing")

    st.sidebar.markdown("---")

    # ========== DATA SETTINGS ==========
    st.sidebar.subheader("📈 Step 2: Data Settings")

    # Use loaded ticker and interval as default if strategy was loaded
    default_ticker = loaded_ticker if loaded_params else "SPY"
    default_interval = loaded_params.get('interval', '1d') if loaded_params else '1d'

    # Make ticker read-only display if strategy is loaded
    if loaded_params:
        st.sidebar.text_input("Ticker Symbol", value=default_ticker, disabled=True,
                              help="Ticker is set by loaded strategy")
        ticker = default_ticker
    else:
        ticker = st.sidebar.text_input("Ticker Symbol", value=default_ticker)

    # Check if loaded strategy has optimization period
    optimization_period = loaded_params.get('optimization_period') if loaded_params else None
    optimization_bars = loaded_params.get('optimization_bars') if loaded_params else None

    period_options = ["1y", "2y", "3y", "5y", "10y"]

    # Default to optimization period if available
    if optimization_period and optimization_period in period_options:
        default_period_index = period_options.index(optimization_period)
    else:
        default_period_index = 1  # Default to 2y

    period = st.sidebar.selectbox(
        "Data Period",
        period_options,
        index=default_period_index,
        help="💡 Testing on different periods than optimization is GOOD - it tests robustness!"
    )

    if loaded_params:
        if optimization_period:
            if period == optimization_period:
                st.sidebar.success(f"✅ Using original optimization period: **{optimization_period}**"
                                  + (f" ({optimization_bars} bars)" if optimization_bars else ""))
            else:
                st.sidebar.warning(f"⚠️ Original optimization: **{optimization_period}**"
                                  + (f" ({optimization_bars} bars)" if optimization_bars else "")
                                  + f"\nCurrently testing: **{period}** (robustness test)")
        else:
            st.sidebar.caption("ℹ️ Optimization period unknown (older strategy). "
                              "Testing on different periods helps verify robustness.")

    interval_options = ["1d", "1h", "15m", "5m"]
    interval_index = interval_options.index(default_interval) if default_interval in interval_options else 0
    if loaded_params:
        st.sidebar.selectbox("Interval", interval_options, index=interval_index, disabled=True,
                            help="Interval is set by loaded strategy")
        interval = default_interval
    else:
        interval = st.sidebar.selectbox("Interval", interval_options, index=0)

    starting_capital = st.sidebar.number_input("Starting Capital ($)", 1000, 10000000, 100000, 1000)

    st.sidebar.markdown("---")

    # ========== STRATEGY PARAMETERS ==========
    st.sidebar.subheader("⚙️ Step 3: Strategy Parameters")

    if loaded_params:
        st.sidebar.success("✅ Using parameters from loaded strategy")

    # Get default values from loaded params or use defaults
    default_signal = loaded_params.get('signal_type', 'any_reversal') if loaded_params else 'any_reversal'
    default_oversold = loaded_params.get('oversold_threshold', -0.3) if loaded_params else -0.3
    default_overbought = loaded_params.get('overbought_threshold', 0.3) if loaded_params else 0.3
    default_sl = loaded_params.get('stop_loss_pct', 3.0) if loaded_params else 3.0
    default_tp = loaded_params.get('take_profit_pct', 8.0) if loaded_params else 8.0
    default_vel_smooth = int(loaded_params.get('vel_smoothing', 3)) if loaded_params else 3
    default_min_bars = int(loaded_params.get('min_bars_between', 5)) if loaded_params else 5
    default_extreme_mult = loaded_params.get('extreme_zone_mult', 1.5) if loaded_params else 1.5
    default_exit_opposite = loaded_params.get('exit_on_opposite_signal', True) if loaded_params else True
    default_exit_midline = loaded_params.get('exit_on_midline_cross', False) if loaded_params else False

    # Advanced filter defaults
    default_rsi_filter = loaded_params.get('rsi_filter', 'none') if loaded_params else 'none'
    default_rsi_period = int(loaded_params.get('rsi_period', 14)) if loaded_params else 14
    default_rsi_oversold = int(loaded_params.get('rsi_oversold', 30)) if loaded_params else 30
    default_rsi_overbought = int(loaded_params.get('rsi_overbought', 70)) if loaded_params else 70
    default_use_macd = loaded_params.get('use_macd_confirm', False) if loaded_params else False
    default_use_bb = loaded_params.get('use_bb_filter', False) if loaded_params else False
    default_require_accel = loaded_params.get('require_accel', False) if loaded_params else False

    signal_type_options = ['velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only',
                           'momentum', 'any_reversal']
    signal_type_index = signal_type_options.index(default_signal) if default_signal in signal_type_options else 4

    signal_type = st.sidebar.selectbox(
        "Signal Type",
        signal_type_options,
        index=signal_type_index,
        key="param_signal_type"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        # Clamp oversold to valid range
        oversold_val = max(-0.6, min(-0.1, float(default_oversold)))
        oversold_threshold = st.slider("Oversold", -0.6, -0.1, oversold_val, 0.01, key="param_oversold")
    with col2:
        # Clamp overbought to valid range
        overbought_val = max(0.1, min(0.6, float(default_overbought)))
        overbought_threshold = st.slider("Overbought", 0.1, 0.6, overbought_val, 0.01, key="param_overbought")

    col3, col4 = st.sidebar.columns(2)
    with col3:
        sl_val = max(0.0, min(10.0, float(default_sl)))
        stop_loss_pct = st.slider("Stop Loss %", 0.0, 10.0, sl_val, 0.1, key="param_sl")
    with col4:
        tp_val = max(0.0, min(20.0, float(default_tp)))
        take_profit_pct = st.slider("Take Profit %", 0.0, 20.0, tp_val, 0.1, key="param_tp")

    vel_smooth_val = max(1, min(10, default_vel_smooth))
    vel_smoothing = st.sidebar.slider("Velocity Smoothing", 1, 10, vel_smooth_val, key="param_vel_smooth")

    min_bars_val = max(1, min(20, default_min_bars))
    min_bars_between = st.sidebar.slider("Min Bars Between", 1, 20, min_bars_val, key="param_min_bars")

    extreme_mult_val = max(1.0, min(3.0, float(default_extreme_mult)))
    extreme_zone_mult = st.sidebar.slider("Extreme Zone Mult", 1.0, 3.0, extreme_mult_val, 0.1, key="param_extreme")

    exit_on_opposite = st.sidebar.checkbox("Exit on Opposite Signal", value=default_exit_opposite, key="param_exit_opp")
    exit_on_midline = st.sidebar.checkbox("Exit on Midline Cross", value=default_exit_midline, key="param_exit_mid")

    # Advanced Filters Section
    with st.sidebar.expander("🎛️ Advanced Filters", expanded=default_rsi_filter != 'none' or default_use_macd or default_use_bb):
        rsi_filter_options = ['none', 'oversold_only', 'overbought_only', 'both']
        rsi_filter_index = rsi_filter_options.index(default_rsi_filter) if default_rsi_filter in rsi_filter_options else 0
        rsi_filter = st.selectbox("RSI Filter", rsi_filter_options, index=rsi_filter_index, key="param_rsi_filter")

        if rsi_filter != 'none':
            rsi_period_val = max(5, min(30, default_rsi_period))
            rsi_period = st.slider("RSI Period", 5, 30, rsi_period_val, key="param_rsi_period")
            col_rsi1, col_rsi2 = st.columns(2)
            with col_rsi1:
                rsi_oversold_val = max(10, min(50, default_rsi_oversold))
                rsi_oversold = st.slider("RSI Oversold", 10, 50, rsi_oversold_val, key="param_rsi_os")
            with col_rsi2:
                rsi_overbought_val = max(50, min(90, default_rsi_overbought))
                rsi_overbought = st.slider("RSI Overbought", 50, 90, rsi_overbought_val, key="param_rsi_ob")
        else:
            rsi_period = default_rsi_period
            rsi_oversold = default_rsi_oversold
            rsi_overbought = default_rsi_overbought

        use_macd_confirm = st.checkbox("Use MACD Confirmation", value=default_use_macd, key="param_macd")
        use_bb_filter = st.checkbox("Use Bollinger Band Filter", value=default_use_bb, key="param_bb")
        require_accel = st.checkbox("Require Positive Acceleration", value=default_require_accel, key="param_accel")

    # Build final params dict
    params = {
        'signal_type': signal_type,
        'oversold_threshold': oversold_threshold,
        'overbought_threshold': overbought_threshold,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'vel_smoothing': vel_smoothing,
        'min_bars_between': min_bars_between,
        'extreme_zone_mult': extreme_zone_mult,
        'exit_on_opposite_signal': exit_on_opposite,
        'exit_on_midline_cross': exit_on_midline,
        'rsi_filter': rsi_filter,
        'rsi_period': rsi_period,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'use_macd_confirm': use_macd_confirm,
        'use_bb_filter': use_bb_filter,
        'require_accel': require_accel
    }

    # Store in session state
    st.session_state['current_params'] = params

    # ========== LOAD DATA ==========
    st.header("1. Load Data & Baseline Backtest")

    if st.button("Load Data", type="primary"):
        with st.spinner(f"Loading {ticker} data..."):
            df = load_data(ticker, period, interval)

            if df is not None and len(df) > 50:
                df = create_composite_oscillator(df)
                st.session_state['test_df'] = df
                st.session_state['test_ticker'] = ticker
                st.session_state['test_params'] = params
                st.session_state['test_interval'] = interval
                st.success(f"Loaded {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                st.error("Failed to load data or insufficient data")

    if 'test_df' not in st.session_state:
        st.info("Load data to begin testing")
        return

    df = st.session_state['test_df']
    ticker = st.session_state.get('test_ticker', 'SPY')
    interval = st.session_state.get('test_interval', '1d')

    # Run baseline backtest
    st.subheader("Baseline Performance")

    baseline_result = run_velocity_backtest(df, params, starting_capital)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{baseline_result['total_return']:.1f}%")
    col2.metric("Trades", baseline_result['num_trades'])
    col3.metric("Win Rate", f"{baseline_result['win_rate']:.1f}%")
    col4.metric("Profit Factor", f"{baseline_result['profit_factor']:.2f}")
    col5.metric("Max Drawdown", f"{baseline_result['max_drawdown']:.1f}%")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Sharpe Ratio", f"{baseline_result['sharpe_ratio']:.2f}")
    col7.metric("Avg Trade", f"{baseline_result['avg_trade']:.2f}%")
    col8.metric("Starting Capital", f"${starting_capital:,.0f}")
    col9.metric("Ending Capital", f"${baseline_result['ending_capital']:,.0f}")
    col10.metric("Profit/Loss", f"${baseline_result['ending_capital'] - starting_capital:+,.0f}")

    st.session_state['baseline_result'] = baseline_result

    st.markdown("---")

    # ========== TESTING SECTIONS ==========

    # Create tabs for different tests
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Walk-Forward", "Monte Carlo", "Sensitivity", "Cross-Asset",
        "Regime Analysis", "Statistical Tests", "Paper Trading"
    ])

    # ========== TAB 1: WALK-FORWARD ANALYSIS ==========
    with tab1:
        st.header("Walk-Forward Analysis")
        st.markdown("""
        Tests strategy on rolling out-of-sample windows to simulate real-world performance.
        Each split trains on historical data and tests on the following period.
        """)

        wf_col1, wf_col2, wf_col3 = st.columns(3)
        with wf_col1:
            n_splits = st.slider("Number of Splits", 3, 10, 5, key="wf_splits")
        with wf_col2:
            train_pct = st.slider("Train Percentage", 0.5, 0.8, 0.7, 0.05, key="wf_train")
        with wf_col3:
            st.metric("Test Percentage", f"{(1-train_pct)*100:.0f}%")

        if st.button("Run Walk-Forward Analysis", type="primary", key="run_wf"):
            with st.spinner("Running walk-forward analysis..."):
                wf_results = run_walk_forward_analysis(df, params, n_splits, train_pct, starting_capital)
                st.session_state['wf_results'] = wf_results

        if 'wf_results' in st.session_state:
            wf_results = st.session_state['wf_results']

            if wf_results['splits']:
                st.subheader("Aggregate Results")
                agg = wf_results['aggregate']

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Return", f"{agg['avg_return']:.1f}%", delta=f"±{agg['std_return']:.1f}%")
                col2.metric("Consistency", f"{agg['consistency']:.0f}%", help="% of profitable periods")
                col3.metric("Avg Win Rate", f"{agg['avg_win_rate']:.1f}%")
                col4.metric("Total Trades", agg['total_trades'])

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Avg Profit Factor", f"{agg['avg_profit_factor']:.2f}")
                col6.metric("Avg Max DD", f"{agg['avg_max_drawdown']:.1f}%")
                col7.metric("Avg Sharpe", f"{agg['avg_sharpe']:.2f}")
                col8.metric("Splits Tested", agg['n_splits'])

                # Show split details
                st.subheader("Split Details")
                split_df = pd.DataFrame(wf_results['splits'])
                split_df['test_start'] = pd.to_datetime(split_df['test_start']).dt.strftime('%Y-%m-%d')
                split_df['test_end'] = pd.to_datetime(split_df['test_end']).dt.strftime('%Y-%m-%d')

                display_cols = ['split', 'test_start', 'test_end', 'test_bars', 'total_return',
                               'num_trades', 'win_rate', 'profit_factor']
                st.dataframe(split_df[display_cols], use_container_width=True)

                # Visualization
                fig = make_subplots(rows=2, cols=1, subplot_titles=('Return by Split', 'Win Rate by Split'))

                fig.add_trace(go.Bar(
                    x=[f"Split {s['split']}" for s in wf_results['splits']],
                    y=[s['total_return'] for s in wf_results['splits']],
                    marker_color=['green' if s['total_return'] > 0 else 'red' for s in wf_results['splits']],
                    name='Return %'
                ), row=1, col=1)

                fig.add_trace(go.Bar(
                    x=[f"Split {s['split']}" for s in wf_results['splits']],
                    y=[s['win_rate'] for s in wf_results['splits']],
                    marker_color='blue',
                    name='Win Rate %'
                ), row=2, col=1)

                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)

                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid splits generated. Try with more data or fewer splits.")

    # ========== TAB 2: MONTE CARLO SIMULATION ==========
    with tab2:
        st.header("Monte Carlo Simulation")
        st.markdown("""
        Randomizes trade order to estimate the distribution of possible outcomes.
        Shows what could happen with the same trades in different sequences.
        """)

        if not baseline_result['trades']:
            st.warning("No trades in baseline result. Run backtest first.")
        else:
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100, key="mc_sims")
            with mc_col2:
                st.metric("Trades to Simulate", len(baseline_result['trades']))

            if st.button("Run Monte Carlo Simulation", type="primary", key="run_mc"):
                with st.spinner(f"Running {n_simulations} simulations..."):
                    mc_results = run_monte_carlo_simulation(
                        baseline_result['trades'], n_simulations, starting_capital
                    )
                    st.session_state['mc_results'] = mc_results

            if 'mc_results' in st.session_state:
                mc_results = st.session_state['mc_results']
                stats = mc_results['statistics']

                st.subheader("Simulation Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Return", f"{stats['mean_return']:.1f}%")
                col2.metric("Median Return", f"{stats['median_return']:.1f}%")
                col3.metric("Std Dev", f"{stats['std_return']:.1f}%")
                col4.metric("Prob of Profit", f"{stats['prob_profit']:.1f}%")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("5th Percentile", f"{stats['percentile_5']:.1f}%", help="Worst 5% of outcomes")
                col6.metric("25th Percentile", f"{stats['percentile_25']:.1f}%")
                col7.metric("75th Percentile", f"{stats['percentile_75']:.1f}%")
                col8.metric("95th Percentile", f"{stats['percentile_95']:.1f}%", help="Best 5% of outcomes")

                st.markdown("---")
                st.subheader("Drawdown Analysis")

                col9, col10, col11 = st.columns(3)
                col9.metric("Mean Max DD", f"{stats['mean_max_dd']:.1f}%")
                col10.metric("Worst Max DD", f"{stats['worst_max_dd']:.1f}%")
                col11.metric("95th Percentile DD", f"{stats['percentile_95_dd']:.1f}%")

                # Distribution plots
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Return Distribution', 'Max Drawdown Distribution'))

                fig.add_trace(go.Histogram(
                    x=mc_results['returns'],
                    nbinsx=50,
                    marker_color='blue',
                    opacity=0.7,
                    name='Returns'
                ), row=1, col=1)

                fig.add_trace(go.Histogram(
                    x=mc_results['max_drawdowns'],
                    nbinsx=50,
                    marker_color='red',
                    opacity=0.7,
                    name='Max DD'
                ), row=1, col=2)

                # Add baseline return line
                fig.add_vline(x=baseline_result['total_return'], line_dash="dash",
                             line_color="green", annotation_text="Actual", row=1, col=1)
                fig.add_vline(x=baseline_result['max_drawdown'], line_dash="dash",
                             line_color="green", annotation_text="Actual", row=1, col=2)

                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(title_text="Return %", row=1, col=1)
                fig.update_xaxes(title_text="Max Drawdown %", row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 3: SENSITIVITY ANALYSIS ==========
    with tab3:
        st.header("Sensitivity Analysis")
        st.markdown("""
        Tests how sensitive results are to parameter changes.
        High sensitivity may indicate overfitting.
        """)

        # Categorize parameters from the loaded strategy
        numeric_params = {}  # Can be tested
        boolean_params = {}  # Cannot be tested (on/off)
        categorical_params = {}  # Cannot be tested (discrete choices)

        for key, value in params.items():
            if isinstance(value, bool):
                boolean_params[key] = value
            elif isinstance(value, str):
                categorical_params[key] = value
            elif isinstance(value, (int, float)) and value != 0:
                numeric_params[key] = value

        # Show parameter breakdown
        st.subheader("📊 Strategy Parameters Overview")

        col_info1, col_info2, col_info3 = st.columns(3)

        with col_info1:
            st.markdown("**✅ Testable (Numeric)**")
            for k, v in numeric_params.items():
                if isinstance(v, float):
                    st.caption(f"• {k}: {v:.4f}")
                else:
                    st.caption(f"• {k}: {v}")

        with col_info2:
            st.markdown("**⚙️ Fixed (Boolean)**")
            for k, v in boolean_params.items():
                status = "✓ ON" if v else "✗ OFF"
                st.caption(f"• {k}: {status}")

        with col_info3:
            st.markdown("**🏷️ Fixed (Categorical)**")
            for k, v in categorical_params.items():
                st.caption(f"• {k}: {v}")

        st.markdown("---")

        # Settings
        sens_col1, sens_col2 = st.columns(2)
        with sens_col1:
            n_steps = st.slider("Steps per Parameter", 3, 9, 5, key="sens_steps")
        with sens_col2:
            perturbation = st.slider("Perturbation Range", 0.1, 0.5, 0.3, 0.05, key="sens_pert")

        # Let user select which parameters to test
        available_params = list(numeric_params.keys())
        default_params = [p for p in ['oversold_threshold', 'overbought_threshold', 'stop_loss_pct',
                                       'take_profit_pct', 'vel_smoothing', 'min_bars_between',
                                       'extreme_zone_mult', 'rsi_period', 'rsi_oversold', 'rsi_overbought']
                         if p in available_params]

        selected_params = st.multiselect(
            "Select parameters to test",
            options=available_params,
            default=default_params,
            help="Choose which numeric parameters to include in sensitivity analysis"
        )

        # Build param_ranges dynamically based on selection
        param_ranges = {}
        for param in selected_params:
            if param in ['vel_smoothing', 'min_bars_between']:
                param_ranges[param] = (0.5, 2.0)  # Wider range for integer params
            else:
                param_ranges[param] = (1 - perturbation, 1 + perturbation)

        if st.button("Run Sensitivity Analysis", type="primary", key="run_sens", disabled=len(selected_params) == 0):
            with st.spinner("Running sensitivity analysis..."):
                sens_results = run_sensitivity_analysis(df, params, param_ranges, n_steps, starting_capital)
                st.session_state['sens_results'] = sens_results

        if 'sens_results' in st.session_state:
            sens_results = st.session_state['sens_results']

            st.subheader("Sensitivity Ranking")
            st.caption("Higher sensitivity = results change more with parameter changes (potential overfit)")

            ranking_df = pd.DataFrame(sens_results['ranked_sensitivity'], columns=['Parameter', 'Sensitivity Score'])
            ranking_df['Sensitivity Score'] = ranking_df['Sensitivity Score'].round(2)

            # Color code the sensitivity scores
            def highlight_sensitivity(val):
                if val > 30:
                    return 'background-color: #ffcccc'  # Red - high sensitivity
                elif val > 15:
                    return 'background-color: #fff3cd'  # Yellow - medium
                else:
                    return 'background-color: #d4edda'  # Green - low/robust

            styled_df = ranking_df.style.applymap(highlight_sensitivity, subset=['Sensitivity Score'])
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("Parameter Impact Charts")
            st.caption("Dashed line = baseline performance. Steep curves = sensitive parameters.")

            # Create sensitivity plots - dynamically size grid
            n_params = len(sens_results['parameters'])
            if n_params == 0:
                st.warning("No parameters were tested.")
            else:
                # Calculate grid size
                n_cols = min(3, n_params)
                n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=list(sens_results['parameters'].keys())
                )

                row, col = 1, 1
                for param_name, param_data in sens_results['parameters'].items():
                    results = param_data['results']

                    fig.add_trace(go.Scatter(
                        x=[r['multiplier'] for r in results],
                        y=[r['total_return'] for r in results],
                        mode='lines+markers',
                        name=param_name,
                        showlegend=False
                    ), row=row, col=col)

                    # Add baseline
                    fig.add_hline(y=sens_results['base_return'], line_dash="dash",
                                 line_color="gray", row=row, col=col)

                    col += 1
                    if col > n_cols:
                        col = 1
                        row += 1

                fig.update_layout(height=250 * n_rows)
                st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 4: CROSS-ASSET TESTING ==========
    with tab4:
        st.header("Cross-Asset Testing")
        st.markdown("""
        Tests the same strategy parameters across multiple assets.
        Good strategies should work across related assets.
        """)

        # Preset ticker groups
        ticker_presets = {
            "Major Indices ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
            "Sector ETFs": ["XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY"],
            "Mega Caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
            "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "Commodities": ["GLD", "SLV", "USO", "UNG"],
            "Bonds": ["TLT", "IEF", "SHY", "LQD", "HYG"]
        }

        preset = st.selectbox("Select Preset", list(ticker_presets.keys()))
        default_tickers = ticker_presets[preset]

        tickers_input = st.text_input("Tickers (comma-separated)", ", ".join(default_tickers))
        tickers = [t.strip().upper() for t in tickers_input.split(",")]

        st.write(f"Testing {len(tickers)} assets: {', '.join(tickers)}")

        if st.button("Run Cross-Asset Testing", type="primary", key="run_cross"):
            cross_results = run_cross_asset_testing(tickers, params, period, starting_capital)
            st.session_state['cross_results'] = cross_results

        if 'cross_results' in st.session_state:
            cross_results = st.session_state['cross_results']
            agg = cross_results['aggregate']

            if agg:
                st.subheader("Aggregate Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Assets Tested", agg['n_assets'])
                col2.metric("Profitable", f"{agg['profitable_assets']} ({agg['pct_profitable']:.0f}%)")
                col3.metric("Avg Return", f"{agg['avg_return']:.1f}%", delta=f"±{agg['std_return']:.1f}%")
                col4.metric("Avg Win Rate", f"{agg['avg_win_rate']:.1f}%")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Best Return", f"{agg['max_return']:.1f}%")
                col6.metric("Worst Return", f"{agg['min_return']:.1f}%")
                col7.metric("Avg Profit Factor", f"{agg['avg_profit_factor']:.2f}")
                col8.metric("Failed Assets", agg['n_failed'])

                # Results table
                st.subheader("Individual Results")
                successful = [r for r in cross_results['results'] if r['status'] == 'success']
                if successful:
                    results_df = pd.DataFrame(successful)
                    results_df = results_df.sort_values('total_return', ascending=False)

                    display_cols = ['ticker', 'total_return', 'num_trades', 'win_rate',
                                   'profit_factor', 'max_drawdown', 'sharpe_ratio']
                    st.dataframe(results_df[display_cols], use_container_width=True)

                    # Visualization
                    fig = go.Figure(go.Bar(
                        x=results_df['ticker'],
                        y=results_df['total_return'],
                        marker_color=['green' if r > 0 else 'red' for r in results_df['total_return']],
                        text=[f"{r:.1f}%" for r in results_df['total_return']],
                        textposition='outside'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(title="Return by Asset", height=400, yaxis_title="Return %")
                    st.plotly_chart(fig, use_container_width=True)

            # Show failed assets
            failed = [r for r in cross_results['results'] if r['status'] == 'failed']
            if failed:
                with st.expander(f"Failed Assets ({len(failed)})"):
                    for f in failed:
                        st.write(f"- {f['ticker']}: {f.get('error', 'Unknown error')}")

    # ========== TAB 5: REGIME ANALYSIS ==========
    with tab5:
        st.header("Market Regime Analysis")
        st.markdown("""
        Analyzes how the strategy performs in different market conditions:
        - **Bull**: Strong uptrend (SMA slope > 5%)
        - **Bear**: Strong downtrend (SMA slope < -5%)
        - **Sideways**: No clear trend
        - **High Volatility**: Elevated volatility without clear trend
        """)

        if st.button("Run Regime Analysis", type="primary", key="run_regime"):
            with st.spinner("Analyzing market regimes..."):
                regime_results = run_regime_analysis(df, params, starting_capital)
                st.session_state['regime_results'] = regime_results

        if 'regime_results' in st.session_state:
            regime_results = st.session_state['regime_results']

            st.subheader("Regime Distribution")
            dist = regime_results['regime_distribution']
            total_bars = sum(dist.values())

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bull Bars", f"{dist.get('bull', 0)} ({dist.get('bull', 0)/total_bars*100:.0f}%)")
            col2.metric("Bear Bars", f"{dist.get('bear', 0)} ({dist.get('bear', 0)/total_bars*100:.0f}%)")
            col3.metric("Sideways Bars", f"{dist.get('sideways', 0)} ({dist.get('sideways', 0)/total_bars*100:.0f}%)")
            col4.metric("High Vol Bars", f"{dist.get('high_volatility', 0)} ({dist.get('high_volatility', 0)/total_bars*100:.0f}%)")

            st.subheader("Performance by Regime")

            regime_data = []
            for regime, data in regime_results['regime_results'].items():
                regime_data.append({
                    'Regime': regime.replace('_', ' ').title(),
                    'Trades': data.get('n_trades', 0),
                    'Total Return': f"{data.get('total_return', 0):.1f}%",
                    'Avg Trade': f"{data.get('avg_trade', 0):.2f}%",
                    'Win Rate': f"{data.get('win_rate', 0):.1f}%",
                    'Best Trade': f"{data.get('best_trade', 0):.2f}%",
                    'Worst Trade': f"{data.get('worst_trade', 0):.2f}%"
                })

            if regime_data:
                st.dataframe(pd.DataFrame(regime_data), use_container_width=True)

                # Visualization
                regimes = list(regime_results['regime_results'].keys())
                returns = [regime_results['regime_results'][r].get('total_return', 0) for r in regimes]

                fig = go.Figure(go.Bar(
                    x=[r.replace('_', ' ').title() for r in regimes],
                    y=returns,
                    marker_color=['green' if r > 0 else 'red' for r in returns],
                    text=[f"{r:.1f}%" for r in returns],
                    textposition='outside'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Return by Market Regime", height=400, yaxis_title="Return %")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for regime analysis")

    # ========== TAB 6: STATISTICAL TESTS ==========
    with tab6:
        st.header("Statistical Significance Testing")
        st.markdown("""
        Tests whether strategy returns are statistically significant or could be due to chance.
        Uses bootstrap resampling and t-tests.
        """)

        if not baseline_result['trades']:
            st.warning("No trades in baseline result. Run backtest first.")
        else:
            n_bootstrap = st.slider("Bootstrap Samples", 500, 5000, 1000, 100, key="stat_boot")

            if st.button("Run Statistical Tests", type="primary", key="run_stats"):
                with st.spinner("Running statistical tests..."):
                    stat_results = calculate_statistical_significance(baseline_result['trades'], n_bootstrap)
                    st.session_state['stat_results'] = stat_results

            if 'stat_results' in st.session_state:
                stat_results = st.session_state['stat_results']

                st.subheader("Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Sample Size", stat_results['n_trades'])
                col2.metric("Mean Return/Trade", f"{stat_results['mean_return']:.2f}%")
                col3.metric("Sharpe Ratio", f"{stat_results['sharpe_ratio']:.2f}")

                st.subheader("Confidence Intervals (95%)")

                mean_ci = stat_results['mean_ci_95']
                sharpe_ci = stat_results['sharpe_ci_95']

                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Mean Return CI", f"[{mean_ci[0]:.2f}%, {mean_ci[1]:.2f}%]")
                    if stat_results['mean_significant']:
                        st.success("Mean is significantly different from zero")
                    else:
                        st.warning("Mean is NOT significantly different from zero")

                with col5:
                    st.metric("Sharpe Ratio CI", f"[{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
                    if stat_results['sharpe_significant']:
                        st.success("Sharpe is significantly different from zero")
                    else:
                        st.warning("Sharpe is NOT significantly different from zero")

                st.subheader("T-Test Results")

                col6, col7, col8 = st.columns(3)
                col6.metric("T-Statistic", f"{stat_results['t_statistic']:.2f}")
                col7.metric("P-Value", f"{stat_results['p_value']:.4f}")

                if stat_results['significant_at_01']:
                    col8.success("Significant at p<0.01")
                elif stat_results['significant_at_05']:
                    col8.warning("Significant at p<0.05")
                else:
                    col8.error("Not statistically significant")

                # Interpretation
                st.markdown("---")
                st.subheader("Interpretation")

                if stat_results['significant_at_05'] and stat_results['mean_significant']:
                    st.success("""
                    **Strong Evidence**: The strategy's positive returns are statistically significant.
                    There's less than a 5% probability these results are due to random chance.
                    """)
                elif stat_results['significant_at_05']:
                    st.warning("""
                    **Moderate Evidence**: The t-test suggests significance, but confidence intervals
                    include zero. Consider gathering more trades for stronger conclusions.
                    """)
                else:
                    st.error("""
                    **Weak Evidence**: Cannot conclude the strategy is better than random.
                    This could be due to:
                    - Insufficient trades
                    - High variance in returns
                    - Strategy may not have real edge
                    """)

    # ========== TAB 7: PAPER TRADING ==========
    with tab7:
        st.header("Paper Trading Mode")
        st.markdown("""
        Real-time signal monitoring and position tracking.
        Shows your current position from the live trading bot.
        """)

        paper_ticker = st.text_input("Ticker for Paper Trading", value=ticker, key="paper_ticker")

        # ===== LOAD CURRENT POSITION FROM LIVE BOT =====
        st.subheader("📍 Current Position (from Live Bot)")

        trade_state_path = "velocity_trade_state.json"
        current_position = None

        if os.path.exists(trade_state_path):
            try:
                with open(trade_state_path, 'r') as f:
                    current_position = json.load(f)
            except:
                current_position = None

        if current_position and current_position.get('position'):
            pos_type = current_position.get('position', 'None').upper()
            entry_price = current_position.get('entry_price', 0)
            entry_time = current_position.get('entry_time', 'Unknown')

            # Get current price for P&L calculation
            try:
                import yfinance as yf
                current_data = yf.download(paper_ticker, period="1d", interval="1d", progress=False)
                if len(current_data) > 0:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(current_data.columns, pd.MultiIndex):
                        current_data.columns = current_data.columns.get_level_values(0)
                    current_price = float(current_data['Close'].iloc[-1])
                else:
                    current_price = float(entry_price)
            except:
                current_price = float(entry_price)

            # Calculate P&L
            if pos_type == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                pnl_color = "green" if pnl_pct >= 0 else "red"
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                pnl_color = "green" if pnl_pct >= 0 else "red"

            # Display position
            pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
            with pos_col1:
                st.markdown(f"### 🟢 {pos_type}")
                st.caption("Active Position")
            with pos_col2:
                st.metric("Entry Price", f"${entry_price:.2f}")
            with pos_col3:
                st.metric("Current Price", f"${current_price:.2f}")
            with pos_col4:
                st.metric("Unrealized P&L", f"{pnl_pct:+.2f}%",
                         delta=f"${(current_price - entry_price):.2f}" if pos_type == 'LONG' else f"${(entry_price - current_price):.2f}")

            st.caption(f"Entry Time: {entry_time}")

            # Check exit conditions
            st.markdown("---")
            st.markdown("**Exit Conditions Check:**")
            exit_col1, exit_col2, exit_col3 = st.columns(3)
            with exit_col1:
                sl_triggered = pnl_pct <= -params.get('stop_loss_pct', 5)
                if sl_triggered:
                    st.error(f"⚠️ STOP LOSS triggered! ({pnl_pct:.1f}% < -{params.get('stop_loss_pct', 5)}%)")
                else:
                    st.success(f"✅ Stop Loss OK ({pnl_pct:.1f}% > -{params.get('stop_loss_pct', 5)}%)")
            with exit_col2:
                tp_triggered = pnl_pct >= params.get('take_profit_pct', 10)
                if tp_triggered:
                    st.success(f"🎯 TAKE PROFIT reached! ({pnl_pct:.1f}% >= {params.get('take_profit_pct', 10)}%)")
                else:
                    st.info(f"📈 Take Profit at {params.get('take_profit_pct', 10)}% (currently {pnl_pct:.1f}%)")
            with exit_col3:
                st.caption(f"Exit on opposite: {'Yes' if params.get('exit_on_opposite_signal') else 'No'}")
                st.caption(f"Exit on midline: {'Yes' if params.get('exit_on_midline_cross') else 'No'}")

        else:
            st.info("📭 **No Active Position** - Waiting for entry signal")

        st.markdown("---")

        # ===== SIGNAL CONDITIONS =====
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Refresh Signal", type="primary", key="refresh_paper"):
                with st.spinner("Loading latest data..."):
                    paper_df = load_data(paper_ticker, "60d", "1d")
                    if paper_df is not None:
                        paper_df = create_composite_oscillator(paper_df)
                        signal = get_paper_trading_signals(paper_df, params)
                        st.session_state['paper_signal'] = signal
                        st.session_state['paper_df'] = paper_df

        if 'paper_signal' in st.session_state:
            signal = st.session_state['paper_signal']

            st.subheader("📊 Current Signal Conditions")

            # Signal display
            signal_color = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'gray'
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"### Signal: :{signal_color[signal['signal']]}[{signal['signal']}]")
                st.caption(f"Strength: {signal['signal_strength'].upper()}")
            with col2:
                st.metric("Price", f"${signal['price']:.2f}")
            with col3:
                st.metric("Timestamp", signal['timestamp'].strftime('%Y-%m-%d %H:%M'))

            st.markdown("---")

            col4, col5, col6, col7 = st.columns(4)
            col4.metric("Oscillator", f"{signal['oscillator']:.3f}")
            col5.metric("Velocity", f"{signal['velocity']:.4f}")
            col6.metric("Acceleration", f"{signal['acceleration']:.4f}")
            col7.metric("Zone", "Oversold" if signal['in_oversold'] else ("Overbought" if signal['in_overbought'] else "Neutral"))

            # Signal conditions
            st.subheader("Signal Conditions")
            conditions = []
            if signal['vel_cross_up']:
                conditions.append("Velocity crossed UP (bullish)")
            if signal['vel_cross_down']:
                conditions.append("Velocity crossed DOWN (bearish)")
            if signal['in_oversold']:
                conditions.append(f"In OVERSOLD zone (< {params['oversold_threshold']})")
            if signal['in_overbought']:
                conditions.append(f"In OVERBOUGHT zone (> {params['overbought_threshold']})")

            if conditions:
                for cond in conditions:
                    st.write(f"- {cond}")
            else:
                st.write("- No active signal conditions")

            # Recent chart
            if 'paper_df' in st.session_state:
                paper_df = st.session_state['paper_df'].tail(50)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.6, 0.4],
                                   subplot_titles=('Price', 'Oscillator'))

                fig.add_trace(go.Candlestick(
                    x=paper_df.index,
                    open=paper_df['open'],
                    high=paper_df['high'],
                    low=paper_df['low'],
                    close=paper_df['close'],
                    name='Price'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=paper_df.index,
                    y=paper_df['composite_smooth'],
                    mode='lines',
                    name='Oscillator',
                    line=dict(color='purple')
                ), row=2, col=1)

                fig.add_hline(y=params['oversold_threshold'], line_dash="dash",
                             line_color="green", row=2, col=1)
                fig.add_hline(y=params['overbought_threshold'], line_dash="dash",
                             line_color="red", row=2, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # ===== PAPER TRADING LOG (PERSISTENT) =====
            st.subheader("📝 Paper Trading Log")

            # Paper trades file path
            paper_trades_file = "paper_trades.json"

            # Load existing paper trades from file
            def load_paper_trades():
                if os.path.exists(paper_trades_file):
                    try:
                        with open(paper_trades_file, 'r') as f:
                            return json.load(f)
                    except:
                        return []
                return []

            def save_paper_trades(trades):
                with open(paper_trades_file, 'w') as f:
                    json.dump(trades, f, indent=2)

            # Initialize paper trades from file
            if 'paper_trades' not in st.session_state:
                st.session_state['paper_trades'] = load_paper_trades()

            # Auto-sync with live bot state
            st.markdown("#### 🔄 Auto-Sync with Live Bot")

            auto_col1, auto_col2 = st.columns([1, 2])
            with auto_col1:
                if st.button("🔄 Sync from Live Bot", help="Check for new trades from the live trading bot"):
                    if current_position and current_position.get('position'):
                        # Check if this position is already logged
                        entry_time = current_position.get('entry_time', '')
                        entry_price = current_position.get('entry_price', 0)

                        # Look for existing open trade with same entry
                        existing_open = None
                        for trade in st.session_state['paper_trades']:
                            if (trade.get('status') == 'OPEN' and
                                trade.get('entry_time') == entry_time):
                                existing_open = trade
                                break

                        if not existing_open:
                            # Log new entry from live bot
                            new_trade = {
                                'id': len(st.session_state['paper_trades']) + 1,
                                'ticker': paper_ticker,
                                'type': current_position.get('position', 'long').upper(),
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'exit_time': None,
                                'exit_price': None,
                                'pnl_pct': None,
                                'pnl_dollars': None,
                                'status': 'OPEN',
                                'source': 'auto_sync',
                                'notes': 'Auto-synced from live bot'
                            }
                            st.session_state['paper_trades'].append(new_trade)
                            save_paper_trades(st.session_state['paper_trades'])
                            st.success(f"✅ Synced OPEN {new_trade['type']} position @ ${entry_price:.2f}")
                            st.rerun()
                        else:
                            st.info("Position already logged")
                    else:
                        # Check if we need to close any open trades
                        closed_count = 0
                        for trade in st.session_state['paper_trades']:
                            if trade.get('status') == 'OPEN' and trade.get('ticker') == paper_ticker:
                                # Position closed - update the trade
                                trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                trade['exit_price'] = signal['price'] if 'paper_signal' in st.session_state else 0
                                if trade['entry_price'] and trade['exit_price']:
                                    if trade['type'] == 'LONG':
                                        trade['pnl_pct'] = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
                                    else:
                                        trade['pnl_pct'] = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100
                                    trade['pnl_dollars'] = (trade['pnl_pct'] / 100) * starting_capital
                                trade['status'] = 'CLOSED'
                                trade['notes'] = (trade.get('notes', '') + ' | Auto-closed on sync').strip(' |')
                                closed_count += 1

                        if closed_count > 0:
                            save_paper_trades(st.session_state['paper_trades'])
                            st.success(f"✅ Closed {closed_count} trade(s)")
                            st.rerun()
                        else:
                            st.info("No open positions to sync")

            with auto_col2:
                st.caption("Syncs entry/exit from `velocity_trade_state.json`")

            st.markdown("---")

            # ===== MANUAL TRADE ENTRY =====
            st.markdown("#### ✏️ Manual Trade Entry")

            with st.expander("➕ Add New Trade", expanded=False):
                man_col1, man_col2, man_col3 = st.columns(3)
                with man_col1:
                    man_type = st.selectbox("Position Type", ["LONG", "SHORT"], key="man_type")
                with man_col2:
                    man_entry_price = st.number_input("Entry Price", value=signal['price'] if 'paper_signal' in st.session_state else 0.0, step=0.01, key="man_entry")
                with man_col3:
                    man_entry_time = st.text_input("Entry Time", value=datetime.now().strftime('%Y-%m-%d %H:%M'), key="man_time")

                man_col4, man_col5 = st.columns(2)
                with man_col4:
                    man_status = st.selectbox("Status", ["OPEN", "CLOSED"], key="man_status")
                with man_col5:
                    man_exit_price = st.number_input("Exit Price (if closed)", value=0.0, step=0.01, key="man_exit", disabled=(man_status == "OPEN"))

                man_notes = st.text_input("Notes", key="man_notes")

                if st.button("💾 Save Trade", key="save_manual_trade"):
                    new_trade = {
                        'id': len(st.session_state['paper_trades']) + 1,
                        'ticker': paper_ticker,
                        'type': man_type,
                        'entry_time': man_entry_time,
                        'entry_price': man_entry_price,
                        'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M') if man_status == "CLOSED" else None,
                        'exit_price': man_exit_price if man_status == "CLOSED" else None,
                        'pnl_pct': None,
                        'pnl_dollars': None,
                        'status': man_status,
                        'source': 'manual',
                        'notes': man_notes
                    }

                    # Calculate P&L if closed
                    if man_status == "CLOSED" and man_entry_price > 0 and man_exit_price > 0:
                        if man_type == "LONG":
                            new_trade['pnl_pct'] = ((man_exit_price - man_entry_price) / man_entry_price) * 100
                        else:
                            new_trade['pnl_pct'] = ((man_entry_price - man_exit_price) / man_entry_price) * 100
                        new_trade['pnl_dollars'] = (new_trade['pnl_pct'] / 100) * starting_capital

                    st.session_state['paper_trades'].append(new_trade)
                    save_paper_trades(st.session_state['paper_trades'])
                    st.success("✅ Trade saved!")
                    st.rerun()

            # ===== CLOSE OPEN POSITION =====
            open_trades = [t for t in st.session_state['paper_trades'] if t.get('status') == 'OPEN']
            if open_trades:
                st.markdown("#### 🔴 Close Open Position")
                with st.expander(f"Close Open Trade ({len(open_trades)} open)", expanded=False):
                    for i, trade in enumerate(open_trades):
                        st.write(f"**{trade['type']}** @ ${trade['entry_price']:.2f} on {trade['entry_time']}")
                        close_col1, close_col2 = st.columns(2)
                        with close_col1:
                            close_price = st.number_input(f"Exit Price", value=signal['price'] if 'paper_signal' in st.session_state else 0.0, step=0.01, key=f"close_price_{trade['id']}")
                        with close_col2:
                            close_notes = st.text_input("Exit Notes", key=f"close_notes_{trade['id']}")

                        if st.button(f"Close Trade #{trade['id']}", key=f"close_btn_{trade['id']}"):
                            # Find and update the trade
                            for t in st.session_state['paper_trades']:
                                if t['id'] == trade['id']:
                                    t['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    t['exit_price'] = close_price
                                    t['status'] = 'CLOSED'
                                    if t['entry_price'] > 0 and close_price > 0:
                                        if t['type'] == 'LONG':
                                            t['pnl_pct'] = ((close_price - t['entry_price']) / t['entry_price']) * 100
                                        else:
                                            t['pnl_pct'] = ((t['entry_price'] - close_price) / t['entry_price']) * 100
                                        t['pnl_dollars'] = (t['pnl_pct'] / 100) * starting_capital
                                    t['notes'] = (t.get('notes', '') + f' | {close_notes}').strip(' |')
                                    break
                            save_paper_trades(st.session_state['paper_trades'])
                            st.success("✅ Trade closed!")
                            st.rerun()

            st.markdown("---")

            # ===== TRADE HISTORY =====
            st.markdown("#### 📊 Trade History")

            if st.session_state['paper_trades']:
                # Convert to DataFrame for display
                trades_df = pd.DataFrame(st.session_state['paper_trades'])

                # Reorder columns for better display
                display_cols = ['id', 'ticker', 'type', 'status', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl_pct', 'source', 'notes']
                display_cols = [c for c in display_cols if c in trades_df.columns]
                trades_df = trades_df[display_cols]

                # Format P&L with color
                def highlight_pnl(val):
                    if pd.isna(val):
                        return ''
                    elif val > 0:
                        return 'background-color: #d4edda; color: #155724'
                    elif val < 0:
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''

                # Style the dataframe
                if 'pnl_pct' in trades_df.columns:
                    styled_df = trades_df.style.applymap(highlight_pnl, subset=['pnl_pct'])
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(trades_df, use_container_width=True)

                # Summary statistics
                closed_trades = [t for t in st.session_state['paper_trades'] if t.get('status') == 'CLOSED' and t.get('pnl_pct') is not None]
                if closed_trades:
                    st.markdown("#### 📈 Performance Summary")
                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                    total_pnl = sum(t['pnl_pct'] for t in closed_trades)
                    winners = [t for t in closed_trades if t['pnl_pct'] > 0]
                    losers = [t for t in closed_trades if t['pnl_pct'] < 0]
                    win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0

                    with sum_col1:
                        st.metric("Total Trades", len(closed_trades))
                    with sum_col2:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with sum_col3:
                        st.metric("Total P&L", f"{total_pnl:+.2f}%",
                                 delta="Profit" if total_pnl > 0 else "Loss")
                    with sum_col4:
                        avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
                        st.metric("Avg P&L/Trade", f"{avg_pnl:+.2f}%")

                # Edit/Delete controls
                st.markdown("---")
                with st.expander("🗑️ Edit / Delete Trades"):
                    delete_id = st.number_input("Trade ID to delete", min_value=1, max_value=len(st.session_state['paper_trades']) if st.session_state['paper_trades'] else 1, step=1, key="delete_id")

                    del_col1, del_col2 = st.columns(2)
                    with del_col1:
                        if st.button("🗑️ Delete Trade", key="delete_trade"):
                            st.session_state['paper_trades'] = [t for t in st.session_state['paper_trades'] if t.get('id') != delete_id]
                            # Re-number IDs
                            for i, t in enumerate(st.session_state['paper_trades']):
                                t['id'] = i + 1
                            save_paper_trades(st.session_state['paper_trades'])
                            st.success(f"Deleted trade #{delete_id}")
                            st.rerun()
                    with del_col2:
                        if st.button("🗑️ Clear ALL Trades", key="clear_all"):
                            st.session_state['paper_trades'] = []
                            save_paper_trades([])
                            st.success("All trades cleared!")
                            st.rerun()
            else:
                st.info("No paper trades logged yet. Use the sync button or add trades manually.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    render_oscillator_predictor_testing_page()
