import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    """Calculate RSI with standardized lowercase column names"""
    close_prices = data.get('close')
    if close_prices is None:
        raise ValueError("Could not find closing prices in data")
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, period):
    """Calculate SMA with standardized lowercase column names"""
    close = data.get('close')
    if close is None:
        raise ValueError("Could not find closing prices in data")
    return close.rolling(window=period).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD with standardized lowercase column names"""
    close = data.get('close')
    if close is None:
        raise ValueError("Could not find closing prices in data")
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands with standardized lowercase column names"""
    close = data.get('close')
    if close is None:
        raise ValueError("Could not find closing prices in data")
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_atr(data, period=14):
    """Calculate ATR with standardized lowercase column names"""
    high = data.get('high')
    low = data.get('low')
    close = data.get('close')
    if high is None or low is None or close is None:
        raise ValueError("Missing required price columns for ATR calculation")
    data['h-l'] = high - low
    data['h-pc'] = np.abs(high - close.shift(1))
    data['l-pc'] = np.abs(low - close.shift(1))
    data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return data['tr'].rolling(window=period).mean()

class Backtester:
    def __init__(self, data, strategy_name, strategy_func, params, starting_capital=10000):
        self.data = data
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.params = params
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.equity_curve = [starting_capital]
        self.trades = []
        self.position = None



    def run(self):
        """
        Runs the backtest.
        """
        strategy_result = self.strategy_func(self.data, self.params)
        # Handle new dictionary return format
        if isinstance(strategy_result, dict):
            signals = strategy_result['signals']
        else:
            signals = strategy_result  # Fallback for old format
        signals = signals.fillna(0) # Safeguard against any NaNs

        for i in range(1, len(self.data)):
            # Buy signal - simple universal position sizing
            if signals.iloc[i] == 1 and self.position is None:
                self.position = {
                    'entry_price': self.data['close'].iloc[i], 
                    'entry_date': self.data.index[i], 
                    'size': self.current_capital / self.data['close'].iloc[i]
                }

            # Sell signal
            elif signals.iloc[i] == -1 and self.position is not None:
                exit_price = self.data['close'].iloc[i]
                profit = (exit_price - self.position['entry_price']) * self.position['size']
                self.current_capital += profit
                position_value = self.position['size'] * self.position['entry_price']
                self.trades.append({
                    'entry_date': self.position['entry_date'],
                    'exit_date': self.data.index[i],
                    'entry_price': self.position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': round(self.position['size'], 4),
                    'position_value': round(position_value, 2),
                    'profit': profit
                })
                self.position = None
            
            # Update equity curve daily
            if self.position:
                # Mark-to-market the open position
                current_value = self.position['size'] * self.data['close'].iloc[i]
                self.equity_curve.append(current_value)
            else:
                self.equity_curve.append(self.current_capital)

    def get_results(self):
        """
        Returns the backtesting results with enhanced metrics.
        """
        # Liquidate any open position at the end of the backtest
        if self.position is not None:
            exit_price = self.data['close'].iloc[-1]
            profit = (exit_price - self.position['entry_price']) * self.position['size']
            self.current_capital += profit
            position_value = self.position['size'] * self.position['entry_price']
            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': self.data.index[-1],
                'entry_price': self.position['entry_price'],
                'exit_price': exit_price,
                'contracts': round(self.position['size'], 4),
                'position_value': round(position_value, 2),
                'profit': profit
            })
            self.position = None

        if not self.trades:
            return pd.DataFrame(), {
                'starting_capital': self.starting_capital,
                'ending_capital': self.starting_capital,
                'total_trades': 0,
                'total_profit': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0
            }

        trade_df = pd.DataFrame(self.trades)
        # This is the critical fix: Use the final, compounded capital to calculate profit.
        # The previous logic was flawed and resulted in a near-zero return for every trial.
        ending_capital = self.current_capital
        total_profit = ending_capital - self.starting_capital
        total_return_pct = (total_profit / self.starting_capital) * 100
        
        wins = trade_df[trade_df['profit'] > 0]['profit'].sum()
        losses = abs(trade_df[trade_df['profit'] < 0]['profit'].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')

        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = abs(drawdown.min() * 100)

        summary = {
            'starting_capital': self.starting_capital,
            'ending_capital': round(ending_capital, 2),
            'total_trades': len(trade_df),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return_pct, 2),
            'win_rate': round((trade_df['profit'] > 0).mean() * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'equity_curve': self.equity_curve
        }
        return trade_df, summary
