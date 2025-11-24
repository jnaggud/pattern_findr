#!/usr/bin/env python3
"""
Comprehensive daily logging system for strategy analysis.
Exports detailed CSV files with all indicators, trades, positions, and strategy parameters.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class DetailedStrategyLogger:
    """
    Logs comprehensive daily strategy data for analysis
    """
    
    def __init__(self, ticker, data, strategy_params, output_dir="/Users/jeffersonduggan/Documents/Pattern_FindR/optimizations"):
        self.ticker = ticker
        self.data = data.copy()
        self.strategy_params = strategy_params.copy()
        self.output_dir = output_dir
        self.daily_log = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.current_position = None
        self.current_capital = 100000  # Starting capital
        self.position_history = []
        self.trade_history = []
        
    def log_daily_data(self, date_idx, row, signals, buy_scores, sell_scores, 
                      active_indicators_on_day, trade_action=None, trade_details=None):
        """
        Log comprehensive daily data
        
        Args:
            date_idx: Index of the current date
            row: Data row for the current day
            signals: Signal value (-1, 0, 1)
            buy_scores: Number of buy indicators
            sell_scores: Number of sell indicators
            active_indicators_on_day: Dict of indicator values
            trade_action: 'BUY', 'SELL', or None
            trade_details: Dict with trade information
        """
        
        # Basic market data
        daily_entry = {
            'date': row.get('date', row.name),
            'ticker': self.ticker,
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'daily_return_pct': ((row['close'] / row['open'] - 1) * 100) if row['open'] > 0 else 0
        }
        
        # Strategy signals and scores
        daily_entry.update({
            'signal': signals,
            'signal_text': 'SELL' if signals == -1 else 'BUY' if signals == 1 else 'HOLD',
            'buy_score': buy_scores,
            'sell_score': sell_scores,
            'buy_threshold': self.strategy_params.get('buy_score_threshold', 1),
            'sell_threshold': self.strategy_params.get('sell_score_threshold', 1),
            'buy_signal_triggered': buy_scores >= self.strategy_params.get('buy_score_threshold', 1),
            'sell_signal_triggered': sell_scores >= self.strategy_params.get('sell_score_threshold', 1)
        })
        
        # Add all indicator values
        for indicator, value in active_indicators_on_day.items():
            daily_entry[f'ind_{indicator}'] = value
            
            # Add threshold information if available
            buy_param = f'{indicator}_buy'
            sell_param = f'{indicator}_sell'
            if buy_param in self.strategy_params:
                daily_entry[f'threshold_{indicator}_buy'] = self.strategy_params[buy_param]
            if sell_param in self.strategy_params:
                daily_entry[f'threshold_{indicator}_sell'] = self.strategy_params[sell_param]
        
        # Position and portfolio information
        daily_entry.update({
            'position_shares': self.current_position['shares'] if self.current_position else 0,
            'position_entry_price': self.current_position['entry_price'] if self.current_position else 0,
            'position_current_value': (self.current_position['shares'] * row['close']) if self.current_position else 0,
            'position_unrealized_pnl': ((self.current_position['shares'] * row['close']) - 
                                      (self.current_position['shares'] * self.current_position['entry_price'])) if self.current_position else 0,
            'position_unrealized_pnl_pct': (((row['close'] / self.current_position['entry_price']) - 1) * 100) if self.current_position else 0,
            'cash_balance': self.current_capital - ((self.current_position['shares'] * self.current_position['entry_price']) if self.current_position else 0),
            'total_portfolio_value': self.current_capital + (((self.current_position['shares'] * row['close']) - 
                                   (self.current_position['shares'] * self.current_position['entry_price'])) if self.current_position else 0)
        })
        
        # Trade information
        if trade_action and trade_details:
            daily_entry.update({
                'trade_action': trade_action,
                'trade_shares': trade_details.get('shares', 0),
                'trade_price': trade_details.get('price', 0),
                'trade_value': trade_details.get('value', 0),
                'trade_realized_pnl': trade_details.get('realized_pnl', 0),
                'trade_realized_pnl_pct': trade_details.get('realized_pnl_pct', 0)
            })
        else:
            daily_entry.update({
                'trade_action': 'HOLD',
                'trade_shares': 0,
                'trade_price': 0,
                'trade_value': 0,
                'trade_realized_pnl': 0,
                'trade_realized_pnl_pct': 0
            })
        
        # Strategy configuration (add once per day for reference)
        daily_entry.update({
            'strategy_signal_persistence': self.strategy_params.get('signal_persistence_days', 1),
            'strategy_trend_filter': self.strategy_params.get('use_trend_filter', False),
            'strategy_position_sizing': self.strategy_params.get('enable_position_sizing', False),
            'strategy_max_position_pct': self.strategy_params.get('max_position_pct', 1.0)
        })
        
        self.daily_log.append(daily_entry)
    
    def update_position(self, action, shares, price, date):
        """Update position tracking"""
        if action == 'BUY':
            if self.current_position:
                # Add to existing position (average price)
                total_shares = self.current_position['shares'] + shares
                total_cost = (self.current_position['shares'] * self.current_position['entry_price']) + (shares * price)
                avg_price = total_cost / total_shares
                self.current_position = {'shares': total_shares, 'entry_price': avg_price}
            else:
                self.current_position = {'shares': shares, 'entry_price': price}
        
        elif action == 'SELL' and self.current_position:
            if shares >= self.current_position['shares']:
                # Full exit
                realized_pnl = (price - self.current_position['entry_price']) * self.current_position['shares']
                self.current_capital += realized_pnl
                self.current_position = None
            else:
                # Partial exit
                realized_pnl = (price - self.current_position['entry_price']) * shares
                self.current_capital += realized_pnl
                self.current_position['shares'] -= shares
        
        # Log trade
        self.trade_history.append({
            'date': date,
            'action': action,
            'shares': shares,
            'price': price,
            'value': shares * price
        })
    
    def export_to_csv(self, trial_number=None, return_pct=None):
        """
        Export all logged data to CSV file
        
        Args:
            trial_number: Optional trial number for filename
            return_pct: Optional return percentage for filename
        """
        if not self.daily_log:
            print("⚠️  No data to export!")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(self.daily_log)
        
        # Generate filename with timestamp and details
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename_parts = [self.ticker, timestamp]
        if trial_number is not None:
            filename_parts.append(f"trial{trial_number}")
        if return_pct is not None:
            filename_parts.append(f"ret{return_pct:.1f}pct")
        
        filename = "_".join(filename_parts) + ".csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        # Also export strategy parameters as JSON
        params_filename = filename.replace('.csv', '_params.json')
        params_filepath = os.path.join(self.output_dir, params_filename)
        
        with open(params_filepath, 'w') as f:
            json.dump(self.strategy_params, f, indent=2, default=str)
        
        print(f"📊 Exported detailed log: {filename}")
        print(f"⚙️  Exported parameters: {params_filename}")
        print(f"📈 Total days logged: {len(df)}")
        print(f"💼 Final portfolio value: ${df['total_portfolio_value'].iloc[-1]:,.2f}")
        
        return filepath

def create_comprehensive_strategy_log(ticker, data, strategy_params, signals, 
                                    buy_scores_series, sell_scores_series, 
                                    trades_df=None, trial_number=None, return_pct=None):
    """
    Create a comprehensive log from strategy execution
    
    Args:
        ticker: Stock ticker
        data: Market data with indicators
        strategy_params: Strategy parameters dict
        signals: Series of signals (-1, 0, 1)
        buy_scores_series: Series of buy scores
        sell_scores_series: Series of sell scores  
        trades_df: DataFrame of trades (optional)
        trial_number: Trial number (optional)
        return_pct: Return percentage (optional)
    """
    
    logger = DetailedStrategyLogger(ticker, data, strategy_params)
    
    # Get all indicator columns
    indicator_columns = [col for col in data.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Process each day
    for i, (_, row) in enumerate(data.iterrows()):
        if i >= len(signals):
            break
        
        # Get indicator values for this day
        indicators_today = {}
        for col in indicator_columns:
            if col in row:
                indicators_today[col] = row[col]
        
        # Get scores for this day
        buy_score = buy_scores_series.iloc[i] if i < len(buy_scores_series) else 0
        sell_score = sell_scores_series.iloc[i] if i < len(sell_scores_series) else 0
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Check for trades on this day
        trade_action = None
        trade_details = None
        
        if trades_df is not None and len(trades_df) > 0:
            day_trades = trades_df[trades_df.index == i]  # Assuming trades are indexed by day
            if len(day_trades) > 0:
                trade = day_trades.iloc[0]
                trade_action = 'BUY' if trade.get('action') == 'buy' else 'SELL'
                trade_details = {
                    'shares': trade.get('shares', 0),
                    'price': trade.get('price', 0),
                    'value': trade.get('value', 0),
                    'realized_pnl': trade.get('profit', 0),
                    'realized_pnl_pct': trade.get('return_pct', 0)
                }
                
                # Update logger's position tracking
                logger.update_position(trade_action, trade_details['shares'], trade_details['price'], row.get('date', i))
        
        # Log this day's data
        logger.log_daily_data(
            date_idx=i,
            row=row,
            signals=signal,
            buy_scores=buy_score,
            sell_scores=sell_score,
            active_indicators_on_day=indicators_today,
            trade_action=trade_action,
            trade_details=trade_details
        )
    
    # Export to CSV
    return logger.export_to_csv(trial_number=trial_number, return_pct=return_pct)

if __name__ == "__main__":
    print("📊 Detailed Strategy Logger - Use create_comprehensive_strategy_log() function")
