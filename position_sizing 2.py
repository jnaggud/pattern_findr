"""
Enhancement #2: Position Sizing & Risk Management
Dynamic position sizing based on signal strength and market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class PositionSizer:
    """
    Dynamic position sizing based on signal strength, volatility, and risk management
    """
    
    def __init__(self, base_capital: float = 100000, max_position_pct: float = 0.20):
        """
        Initialize position sizer
        
        Args:
            base_capital: Starting capital
            max_position_pct: Maximum percentage of capital per position (20% = conservative)
        """
        self.base_capital = base_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = 0.01  # Minimum 1% position
        
        # Risk management parameters
        self.max_total_exposure = 0.80  # Maximum 80% of capital deployed
        self.volatility_adjustment = True
        self.kelly_optimization = True
        
    def calculate_signal_strength(self, signal_scores: pd.Series, threshold: float, 
                                  active_indicators: int) -> pd.Series:
        """
        Calculate signal strength as confidence metric
        
        Args:
            signal_scores: Raw signal scores from indicators
            threshold: Threshold used for signal generation
            active_indicators: Total number of active indicators
        
        Returns:
            Signal strength from 0.0 to 2.0 (0.5 = normal, 2.0 = extremely strong)
        """
        # Normalize scores relative to threshold and maximum possible
        max_possible_score = active_indicators * 2.0  # Assume max 2x signal persistence
        
        # Calculate strength relative to threshold
        excess_over_threshold = (signal_scores - threshold).clip(lower=0)
        normalized_excess = excess_over_threshold / max(threshold, 1.0)
        
        # Signal strength: 0.5 (barely triggered) to 2.0 (extremely strong)
        strength = 0.5 + (normalized_excess * 1.5).clip(upper=1.5)
        
        return strength
    
    def calculate_volatility_multiplier(self, data: pd.DataFrame, 
                                       lookback_days: int = 20) -> pd.Series:
        """
        Calculate volatility-based position adjustment
        
        Args:
            data: Market data with 'close' prices
            lookback_days: Period for volatility calculation
        
        Returns:
            Multiplier from 0.5 (high vol) to 1.5 (low vol)
        """
        if 'close' not in data.columns:
            return pd.Series(1.0, index=data.index)
        
        # Calculate rolling volatility (annualized)
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=lookback_days, min_periods=5).std() * np.sqrt(252)
        
        # Calculate volatility percentiles
        vol_median = volatility.rolling(window=60, min_periods=20).median()
        vol_ratio = volatility / vol_median
        
        # High volatility = reduce positions, Low volatility = increase positions
        # Ratio > 1.5 = high vol (0.5x), Ratio < 0.7 = low vol (1.3x)
        multiplier = pd.Series(1.0, index=data.index)
        
        high_vol_mask = vol_ratio > 1.5
        multiplier.loc[high_vol_mask] = 0.5
        
        low_vol_mask = vol_ratio < 0.7
        multiplier.loc[low_vol_mask] = 1.3
        
        # Smooth transition for medium volatility
        medium_vol_mask = (vol_ratio >= 0.7) & (vol_ratio <= 1.5)
        multiplier.loc[medium_vol_mask] = 1.5 - vol_ratio.loc[medium_vol_mask]
        
        return multiplier.fillna(1.0).clip(0.3, 2.0)
    
    def calculate_kelly_criterion(self, historical_trades: pd.DataFrame) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            historical_trades: DataFrame with 'pnl_pct' column
        
        Returns:
            Kelly fraction (optimal position size fraction)
        """
        if historical_trades.empty or 'pnl_pct' not in historical_trades.columns:
            return 0.10  # Default 10% if no historical data
        
        pnl = historical_trades['pnl_pct'] / 100.0  # Convert to decimal
        
        # Calculate win rate and average win/loss
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.05  # Conservative if no wins or no losses
        
        win_rate = len(wins) / len(pnl)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Limit Kelly to reasonable range (2% to 25%)
        kelly_fraction = np.clip(kelly_fraction, 0.02, 0.25)
        
        return kelly_fraction
    
    def calculate_position_sizes(self, signals: pd.Series, signal_scores_buy: pd.Series, 
                                 signal_scores_sell: pd.Series, data: pd.DataFrame,
                                 params: Dict, historical_trades: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate dynamic position sizes for each signal
        
        Args:
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            signal_scores_buy: Raw buy signal scores
            signal_scores_sell: Raw sell signal scores
            data: Market data
            params: Strategy parameters
            historical_trades: Historical performance for Kelly calculation
        
        Returns:
            Position sizes as percentage of capital (0.01 to max_position_pct)
        """
        position_sizes = pd.Series(0.0, index=signals.index)
        
        # Get strategy parameters
        buy_threshold = params.get('buy_score_threshold', 5)
        sell_threshold = params.get('sell_score_threshold', 5)
        active_indicators = len([k for k in params.keys() 
                               if k.startswith('use_') and params[k] is True])
        
        # Calculate signal strengths
        buy_strength = self.calculate_signal_strength(
            signal_scores_buy, buy_threshold, active_indicators
        )
        sell_strength = self.calculate_signal_strength(
            signal_scores_sell, sell_threshold, active_indicators
        )
        
        # Calculate volatility adjustment
        volatility_multiplier = self.calculate_volatility_multiplier(data)
        
        # Calculate Kelly-optimized base position size
        if self.kelly_optimization and historical_trades is not None:
            kelly_fraction = self.calculate_kelly_criterion(historical_trades)
        else:
            kelly_fraction = 0.10  # Default 10%
        
        # Base position sizing
        base_position_pct = min(kelly_fraction, self.max_position_pct)
        
        # Calculate position sizes for buy signals
        buy_signals = signals == 1
        if buy_signals.any():
            # Scale by signal strength and volatility
            buy_positions = (base_position_pct * 
                           buy_strength.loc[buy_signals] * 
                           volatility_multiplier.loc[buy_signals])
            position_sizes.loc[buy_signals] = buy_positions
        
        # Calculate position sizes for sell signals (typically smaller for shorts)
        sell_signals = signals == -1
        if sell_signals.any():
            # Sell positions are typically more conservative
            sell_positions = (base_position_pct * 0.7 *  # 30% reduction for shorts
                            sell_strength.loc[sell_signals] * 
                            volatility_multiplier.loc[sell_signals])
            position_sizes.loc[sell_signals] = sell_positions
        
        # Apply position limits
        position_sizes = position_sizes.clip(
            lower=self.min_position_pct, 
            upper=self.max_position_pct
        )
        
        # Ensure total exposure doesn't exceed maximum
        position_sizes = self._apply_exposure_limits(position_sizes, signals)
        
        return position_sizes
    
    def _apply_exposure_limits(self, position_sizes: pd.Series, signals: pd.Series) -> pd.Series:
        """
        Apply maximum total exposure limits
        """
        # Calculate cumulative exposure (simplified - assumes no overlapping positions)
        active_positions = position_sizes[signals != 0]
        
        if active_positions.empty:
            return position_sizes
        
        total_exposure = active_positions.sum()
        
        # If total exposure exceeds maximum, scale down proportionally
        if total_exposure > self.max_total_exposure:
            scale_factor = self.max_total_exposure / total_exposure
            position_sizes = position_sizes * scale_factor
        
        return position_sizes
    
    def get_risk_metrics(self, position_sizes: pd.Series, data: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics for the position sizing strategy
        """
        active_positions = position_sizes[position_sizes > 0]
        
        if active_positions.empty:
            return {
                'avg_position_size': 0,
                'max_position_size': 0,
                'total_exposure': 0,
                'position_count': 0,
                'risk_level': 'No Positions'
            }
        
        metrics = {
            'avg_position_size': active_positions.mean() * 100,  # As percentage
            'max_position_size': active_positions.max() * 100,
            'total_exposure': active_positions.sum() * 100,
            'position_count': len(active_positions),
            'volatility_adjusted': len(position_sizes[position_sizes != position_sizes.median()]),
        }
        
        # Risk level assessment
        if metrics['total_exposure'] > 60:
            metrics['risk_level'] = 'High'
        elif metrics['total_exposure'] > 30:
            metrics['risk_level'] = 'Medium'
        else:
            metrics['risk_level'] = 'Conservative'
        
        return metrics

def enhance_backtester_with_position_sizing(signals: pd.Series, data: pd.DataFrame, 
                                           params: Dict, capital: float = 100000,
                                           historical_trades: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, Dict]:
    """
    Enhance signals with dynamic position sizing
    
    Returns:
        Tuple of (position_sizes, risk_metrics)
    """
    # This is a placeholder - we'll integrate this with the actual backtester
    # For now, simulate signal scores
    signal_scores_buy = pd.Series(params.get('buy_score_threshold', 5), index=data.index)
    signal_scores_sell = pd.Series(params.get('sell_score_threshold', 5), index=data.index)
    
    # Create position sizer
    sizer = PositionSizer(base_capital=capital, max_position_pct=0.20)
    
    # Calculate position sizes
    position_sizes = sizer.calculate_position_sizes(
        signals, signal_scores_buy, signal_scores_sell, data, params, historical_trades
    )
    
    # Get risk metrics
    risk_metrics = sizer.get_risk_metrics(position_sizes, data)
    
    return position_sizes, risk_metrics

def optimize_position_sizing_parameters(data: pd.DataFrame, signals: pd.Series, 
                                       params: Dict) -> Dict:
    """
    Optimize position sizing parameters for Enhancement #2
    """
    print("🎯 Optimizing position sizing parameters...")
    
    # Test different maximum position sizes
    max_position_tests = [0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}
    
    for max_pos in max_position_tests:
        sizer = PositionSizer(max_position_pct=max_pos)
        
        # Simulate signal scores (in real implementation, these come from universal_strategy)
        signal_scores_buy = pd.Series(params.get('buy_score_threshold', 5), index=data.index)
        signal_scores_sell = pd.Series(params.get('sell_score_threshold', 5), index=data.index)
        
        position_sizes = sizer.calculate_position_sizes(
            signals, signal_scores_buy, signal_scores_sell, data, params
        )
        
        risk_metrics = sizer.get_risk_metrics(position_sizes, data)
        
        results[f"max_pos_{max_pos:.0%}"] = {
            'avg_position': risk_metrics['avg_position_size'],
            'max_position': risk_metrics['max_position_size'],
            'total_exposure': risk_metrics['total_exposure'],
            'risk_level': risk_metrics['risk_level']
        }
    
    # Find optimal balance of risk and return potential
    optimal_config = max(results.keys(), 
                        key=lambda k: results[k]['total_exposure'] if results[k]['risk_level'] != 'High' else 0)
    
    return {
        'optimization_results': results,
        'optimal_config': optimal_config,
        'recommended_max_position': float(optimal_config.split('_')[2].replace('%', '')) / 100,
        'enhancement_summary': {
            'description': 'Dynamic position sizing based on signal strength and volatility',
            'benefits': [
                'Higher returns on high-confidence signals',
                'Reduced risk on marginal signals', 
                'Volatility-aware position adjustments',
                'Kelly Criterion optimization'
            ]
        }
    }
