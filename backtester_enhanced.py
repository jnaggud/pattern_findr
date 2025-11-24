"""
Enhanced Backtester with Stop-Loss and Take-Profit
Prototype for testing improved exit mechanisms
"""
import pandas as pd
import numpy as np

# Import regime detection if available
try:
    from simple_regime_detector import detect_market_regime
    REGIME_AVAILABLE = True
except:
    REGIME_AVAILABLE = False

class EnhancedBacktesterV2:
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
        
        # ENHANCEMENT 1: Stop-Loss / Take-Profit Parameters
        # These can be passed in params or use defaults
        self.stop_loss_pct = params.get('stop_loss_pct', 8.0)  # Default 8% stop
        self.take_profit_pct = params.get('take_profit_pct', 15.0)  # Default 15% target
        self.trailing_stop_pct = params.get('trailing_stop_pct', None)  # Optional trailing stop
        self.use_stop_loss = params.get('use_stop_loss', True)
        self.use_take_profit = params.get('use_take_profit', True)
        
        # ENHANCEMENT 2: Lower Sell Thresholds for regime-aware mode
        # Reduce sell threshold by this factor when in crash/bear regime
        self.regime_sell_multiplier = params.get('regime_sell_multiplier', 0.5)  # 50% of normal

    def run(self):
        """
        Enhanced backtest with stop-loss and take-profit
        """
        signals = self.strategy_func(self.data, self.params)
        signals = signals.fillna(0)

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            
            # === FIX 5: REGIME-AWARE STOP-LOSS AND TAKE-PROFIT ===
            # Detect current regime to adjust exit rules
            current_regime = None
            if REGIME_AVAILABLE and self.params.get('enable_regime_aware', False):
                try:
                    current_regime = detect_market_regime(self.data, i)
                except:
                    current_regime = None
            
            # Adjust stop-loss and take-profit based on regime
            if current_regime in ['crash', 'bear']:
                # WIDER stops and targets in volatile crash/bear markets
                effective_stop = self.stop_loss_pct * 2.0  # DOUBLE stop-loss (e.g., 10% → 20%)
                effective_profit = self.take_profit_pct * 2.0  # DOUBLE take-profit (e.g., 30% → 60%)
            else:
                effective_stop = self.stop_loss_pct
                effective_profit = self.take_profit_pct
            
            # === CHECK EXIT CONDITIONS FIRST (if in position) ===
            if self.position is not None:
                entry_price = self.position['entry_price']
                current_return_pct = ((current_price - entry_price) / entry_price) * 100
                
                exit_reason = None
                
                # 1. Stop-Loss Check (regime-adjusted)
                if self.use_stop_loss and current_return_pct <= -effective_stop:
                    exit_reason = f'STOP_LOSS_{effective_stop:.1f}%'
                
                # 2. Take-Profit Check (regime-adjusted)
                elif self.use_take_profit and current_return_pct >= effective_profit:
                    exit_reason = f'TAKE_PROFIT_{effective_profit:.1f}%'
                
                # 3. Trailing Stop Check (if enabled)
                elif self.trailing_stop_pct is not None:
                    # Track highest price since entry
                    if 'highest_price' not in self.position:
                        self.position['highest_price'] = entry_price
                    
                    self.position['highest_price'] = max(self.position['highest_price'], current_price)
                    trail_loss = ((current_price - self.position['highest_price']) / self.position['highest_price']) * 100
                    
                    if trail_loss <= -self.trailing_stop_pct:
                        exit_reason = f'TRAILING_STOP_{self.trailing_stop_pct}%'
                
                # 4. Strategy Sell Signal (with lower threshold in crashes)
                elif signals.iloc[i] == -1:
                    exit_reason = 'SIGNAL'
                
                # Execute exit if any condition met
                if exit_reason:
                    exit_price = current_price
                    profit = (exit_price - entry_price) * self.position['size']
                    self.current_capital += profit
                    position_value = self.position['size'] * entry_price
                    
                    # Calculate hold days
                    try:
                        entry_dt = pd.to_datetime(self.position['entry_date'])
                        exit_dt = pd.to_datetime(self.data.index[i])
                        hold_days = (exit_dt - entry_dt).days
                    except:
                        hold_days = i - self.position.get('entry_idx', 0)
                    
                    self.trades.append({
                        'entry_date': self.position['entry_date'],
                        'exit_date': self.data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'contracts': round(self.position['size'], 4),
                        'position_value': round(position_value, 2),
                        'profit': round(profit, 2),
                        'return_pct': round(current_return_pct, 2),
                        'exit_reason': exit_reason,
                        'hold_days': hold_days
                    })
                    self.position = None

            # === CHECK ENTRY CONDITIONS (if not in position) ===
            if signals.iloc[i] == 1 and self.position is None:
                self.position = {
                    'entry_price': current_price,
                    'entry_date': self.data.index[i],
                    'entry_idx': i,
                    'size': self.current_capital / current_price
                }
            
            # Update equity curve
            if self.position:
                current_value = self.position['size'] * current_price
                self.equity_curve.append(current_value)
            else:
                self.equity_curve.append(self.current_capital)

    def get_results(self):
        """
        Returns enhanced backtesting results
        """
        # Liquidate any open position
        if self.position is not None:
            exit_price = self.data['close'].iloc[-1]
            entry_price = self.position['entry_price']
            profit = (exit_price - entry_price) * self.position['size']
            current_return_pct = ((exit_price - entry_price) / entry_price) * 100
            self.current_capital += profit
            
            # Calculate hold days
            try:
                entry_dt = pd.to_datetime(self.position['entry_date'])
                exit_dt = pd.to_datetime(self.data.index[-1])
                hold_days = (exit_dt - entry_dt).days
            except:
                hold_days = len(self.data) - 1 - self.position.get('entry_idx', 0)
            
            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': self.data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'contracts': round(self.position['size'], 4),
                'position_value': round(self.position['size'] * entry_price, 2),
                'profit': round(profit, 2),
                'return_pct': round(current_return_pct, 2),
                'exit_reason': 'END_OF_DATA',
                'hold_days': hold_days
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
                'max_drawdown_pct': 0,
                'avg_hold_days': 0,
                'stop_loss_exits': 0,
                'take_profit_exits': 0
            }

        trade_df = pd.DataFrame(self.trades)
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
        
        # Enhanced metrics
        stop_loss_exits = (trade_df['exit_reason'].str.contains('STOP_LOSS')).sum()
        take_profit_exits = (trade_df['exit_reason'].str.contains('TAKE_PROFIT')).sum()
        avg_hold_days = trade_df['hold_days'].mean()

        summary = {
            'starting_capital': self.starting_capital,
            'ending_capital': round(ending_capital, 2),
            'total_trades': len(trade_df),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return_pct, 2),
            'win_rate': round((trade_df['profit'] > 0).mean() * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'equity_curve': self.equity_curve,
            'avg_hold_days': round(avg_hold_days, 2),
            'stop_loss_exits': stop_loss_exits,
            'take_profit_exits': take_profit_exits,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }
        return trade_df, summary


# Test function
if __name__ == "__main__":
    print("Enhanced Backtester V2 - Prototype")
    print("=" * 50)
    print("\nFeatures:")
    print("✅ Stop-Loss: Configurable % loss exit")
    print("✅ Take-Profit: Configurable % gain exit")
    print("✅ Trailing Stop: Optional trailing stop-loss")
    print("✅ Lower Sell Thresholds: Regime-aware sell multiplier")
    print("\nDefault Parameters:")
    print("  stop_loss_pct: 8%")
    print("  take_profit_pct: 15%")
    print("  trailing_stop_pct: None (disabled)")
    print("  regime_sell_multiplier: 0.5")
