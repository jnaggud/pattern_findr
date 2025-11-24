"""
Enhanced Backtester with Dynamic Position Sizing for Enhancement #2
"""

import pandas as pd
import numpy as np
from backtester import Backtester
from position_sizing import PositionSizer
from typing import Dict, Optional

class EnhancedBacktester(Backtester):
    """
    Enhanced backtester with dynamic position sizing capabilities
    """
    
    def __init__(self, data, strategy_name, strategy_func, params, starting_capital=100000, 
                 enable_position_sizing=True, max_position_pct=0.20):
        """
        Initialize enhanced backtester
        
        Args:
            enable_position_sizing: Whether to use dynamic position sizing
            max_position_pct: Maximum position size as percentage of capital
        """
        super().__init__(data, strategy_name, strategy_func, params, starting_capital)
        
        # Enhancement #2: Position sizing configuration
        self.enable_position_sizing = enable_position_sizing
        self.position_sizer = PositionSizer(
            base_capital=starting_capital,
            max_position_pct=max_position_pct
        ) if enable_position_sizing else None
        
        # Track signal scores for position sizing
        self.signal_scores_buy = None
        self.signal_scores_sell = None
        self.position_sizes = None
        
        # Enhanced tracking
        self.total_deployed_capital = 0.0
        self.position_sizing_metrics = {}
    
    def run(self):
        """
        Enhanced backtest with dynamic position sizing
        """
        # Get signals from strategy
        signals = self.strategy_func(self.data, self.params)
        signals = signals.fillna(0)
        
        # Enhancement #2: Calculate dynamic position sizes if enabled
        if self.enable_position_sizing and self.position_sizer:
            self.position_sizes = self._calculate_enhanced_position_sizes(signals)
        else:
            # Default: use full capital for each position (original behavior)
            self.position_sizes = pd.Series(1.0, index=signals.index)
        
        # Run enhanced backtest loop
        self._run_enhanced_backtest(signals)
        
        # Calculate position sizing metrics
        if self.enable_position_sizing:
            self.position_sizing_metrics = self.position_sizer.get_risk_metrics(
                self.position_sizes, self.data
            )
    
    def _calculate_enhanced_position_sizes(self, signals: pd.Series) -> pd.Series:
        """
        Calculate AGGRESSIVE enhanced position sizes for higher returns
        """
        max_pos_pct = self.params.get('max_position_pct', 0.50)  # More aggressive default
        volatility_adjustment = self.params.get('volatility_adjustment', True)
        aggressive_vol_scaling = self.params.get('aggressive_vol_scaling', False)
        confidence_risk_multiplier = self.params.get('confidence_risk_multiplier', 1.5)
        
        # Start with base position size
        position_sizes = pd.Series(0.0, index=signals.index)
        
        # Calculate volatility multiplier
        if volatility_adjustment and 'close' in self.data.columns:
            returns = self.data['close'].pct_change()
            volatility = returns.rolling(window=20, min_periods=5).std().fillna(0.02)
            vol_median = volatility.median()
            vol_multiplier = pd.Series(1.0, index=signals.index)
            
            if aggressive_vol_scaling:
                # AGGRESSIVE: Less conservative volatility adjustments
                high_vol_mask = volatility > vol_median * 2.0  # Only reduce on extreme vol
                vol_multiplier.loc[high_vol_mask] = 0.8  # Less reduction
                
                low_vol_mask = volatility < vol_median * 0.5  # More aggressive on low vol
                vol_multiplier.loc[low_vol_mask] = 1.5  # Higher increase
            else:
                # Standard volatility adjustments  
                high_vol_mask = volatility > vol_median * 1.5
                vol_multiplier.loc[high_vol_mask] = 0.7
                
                low_vol_mask = volatility < vol_median * 0.7
                vol_multiplier.loc[low_vol_mask] = 1.2
        else:
            vol_multiplier = pd.Series(1.0, index=signals.index)
        
        # AGGRESSIVE: Signal strength with higher variance (0.6x to 2.0x)
        signal_mask = signals != 0
        np.random.seed(42)
        
        # More aggressive signal strength simulation
        if confidence_risk_multiplier > 1.3:
            # High confidence multiplier = more aggressive sizing
            signal_strength = 0.6 + 1.4 * np.random.random(len(signals))  # 0.6x to 2.0x
        else:
            # Standard signal strength
            signal_strength = 0.8 + 0.4 * np.random.random(len(signals))  # 0.8x to 1.2x
        
        strength_series = pd.Series(signal_strength, index=signals.index)
        
        # Apply confidence risk multiplier
        strength_series = strength_series * confidence_risk_multiplier
        
        # Calculate aggressive position sizes
        base_position_sizes = max_pos_pct * vol_multiplier * strength_series
        
        # AGGRESSIVE: Allow positions up to 150% of max_pos_pct for extreme confidence
        position_sizes = base_position_sizes.clip(lower=0.01, upper=max_pos_pct * 1.5)
        
        # Only apply to actual signals
        position_sizes.loc[~signal_mask] = 0.0
        
        return position_sizes
    
    def _run_enhanced_backtest(self, signals: pd.Series):
        """
        Run backtest with dynamic position sizing
        """
        for i in range(1, len(self.data)):
            current_signal = signals.iloc[i]
            current_price = self.data['close'].iloc[i]
            
            # Buy signal with dynamic position sizing
            if current_signal == 1 and self.position is None:
                position_size_pct = self.position_sizes.iloc[i]
                position_capital = self.current_capital * position_size_pct
                shares = position_capital / current_price
                
                self.position = {
                    'entry_price': current_price,
                    'entry_date': self.data.index[i],
                    'size': shares,
                    'position_capital': position_capital,
                    'position_size_pct': position_size_pct
                }
                
                # Track deployed capital
                self.total_deployed_capital += position_capital
                
                print(f"📈 BUY: {shares:.2f} shares @ ${current_price:.2f} "
                      f"(${position_capital:.0f} = {position_size_pct:.1%} of capital)")
            
            # Sell signal
            elif current_signal == -1 and self.position is not None:
                exit_price = current_price
                profit = (exit_price - self.position['entry_price']) * self.position['size']
                
                # Calculate returns
                position_return_pct = (exit_price - self.position['entry_price']) / self.position['entry_price'] * 100
                capital_impact = profit / self.starting_capital * 100
                
                self.current_capital += profit
                
                self.trades.append({
                    'entry_date': self.position['entry_date'],
                    'exit_date': self.data.index[i],
                    'entry_price': self.position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': round(self.position['size'], 4),
                    'position_value': round(self.position['position_capital'], 2),
                    'position_size_pct': round(self.position['position_size_pct'] * 100, 1),
                    'profit': profit,
                    'position_return_pct': round(position_return_pct, 2),
                    'capital_impact_pct': round(capital_impact, 2)
                })
                
                print(f"📉 SELL: {self.position['size']:.2f} shares @ ${exit_price:.2f} "
                      f"(${profit:+.0f} = {position_return_pct:+.1f}% position return)")
                
                self.position = None
            
            # Update equity curve with mark-to-market
            if self.position:
                current_position_value = self.position['size'] * current_price
                unrealized_pnl = current_position_value - self.position['position_capital']
                total_equity = self.current_capital + unrealized_pnl
                self.equity_curve.append(total_equity)
            else:
                self.equity_curve.append(self.current_capital)
    
    def get_results(self):
        """
        Enhanced results with position sizing metrics
        """
        trade_df, summary = super().get_results()
        
        # Add Enhancement #2 metrics
        if self.enable_position_sizing and self.position_sizing_metrics:
            summary.update({
                'position_sizing_enabled': True,
                'avg_position_size_pct': self.position_sizing_metrics.get('avg_position_size', 0),
                'max_position_size_pct': self.position_sizing_metrics.get('max_position_size', 0),
                'total_exposure_pct': self.position_sizing_metrics.get('total_exposure', 0),
                'risk_level': self.position_sizing_metrics.get('risk_level', 'Unknown'),
                'position_sizing_efficiency': self._calculate_efficiency_metric(trade_df)
            })
        else:
            summary.update({
                'position_sizing_enabled': False,
                'avg_position_size_pct': 100.0,  # Full capital deployment
                'max_position_size_pct': 100.0,
                'total_exposure_pct': 100.0,
                'risk_level': 'High',
                'position_sizing_efficiency': 0
            })
        
        return trade_df, summary
    
    def _calculate_efficiency_metric(self, trade_df: pd.DataFrame) -> float:
        """
        Calculate position sizing efficiency metric
        """
        if trade_df.empty or 'position_size_pct' not in trade_df.columns:
            return 0
        
        # Efficiency = correlation between position size and trade performance
        position_sizes = trade_df['position_size_pct']
        trade_returns = trade_df['position_return_pct']
        
        if len(position_sizes) < 2:
            return 0
        
        correlation = np.corrcoef(position_sizes, trade_returns)[0, 1]
        return round(correlation * 100, 1) if not np.isnan(correlation) else 0

def create_enhanced_backtester(data, strategy_name, strategy_func, params, 
                             starting_capital=100000, enhancement_level=2):
    """
    Factory function to create appropriate backtester based on enhancement level
    
    Args:
        enhancement_level: 1=basic, 2=position sizing, 3=regime adaptation, etc.
    """
    if enhancement_level >= 2:
        return EnhancedBacktester(
            data=data,
            strategy_name=strategy_name,
            strategy_func=strategy_func,
            params=params,
            starting_capital=starting_capital,
            enable_position_sizing=True,
            max_position_pct=0.20  # Conservative starting point
        )
    else:
        return Backtester(data, strategy_name, strategy_func, params, starting_capital)
