"""
Enhanced Trading Engine with Smart Entry Timing and Dynamic Risk Management
Addresses the issues:
1. Better entry timing (not just first signal)
2. Confidence-based signal filtering  
3. Dynamic stop loss and take profit
4. Missing signal detection and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dynamic_risk_manager import DynamicRiskManager, EnhancedSignalFilter

class EnhancedTradingEngine:
    def __init__(self, 
                 min_confidence_threshold: float = 0.35,
                 min_convergence_score: float = 0.2,
                 signal_patience_days: int = 3):
        """
        Enhanced trading engine with smart signal filtering
        
        Args:
            min_confidence_threshold: Minimum ML confidence to consider trade
            min_convergence_score: Minimum ML/Technical convergence required
            signal_patience_days: Days to wait for better signal confirmation
        """
        self.min_confidence = min_confidence_threshold
        self.min_convergence = min_convergence_score  
        self.patience_days = signal_patience_days
        
        self.risk_manager = DynamicRiskManager()
        self.signal_filter = EnhancedSignalFilter()
        
        # Signal quality tracking
        self.signal_history = []
        self.missed_opportunities = []
        
    def evaluate_trading_opportunity(self,
                                   current_date: pd.Timestamp,
                                   signal: int,
                                   ml_confidence: float,
                                   composite_tech: float,
                                   price: float,
                                   price_history: pd.DataFrame,
                                   ml_history: pd.Series,
                                   tech_history: pd.Series,
                                   volume_history: pd.Series = None) -> Dict:
        """
        Evaluate whether to enter a trade based on signal quality and timing
        
        Returns comprehensive analysis of trading opportunity
        """
        
        # 1. Basic signal filtering
        filter_result = self.signal_filter.should_enter_trade(
            signal, ml_confidence, composite_tech, ml_history, tech_history
        )
        
        # 2. Calculate dynamic risk levels
        risk_levels = self.risk_manager.calculate_dynamic_levels(
            current_price=price,
            ml_confidence=ml_confidence,
            composite_tech=composite_tech,
            signal_direction=signal,
            price_data=price_history,
            volume_data=volume_history
        )
        
        # 3. Analyze signal timing quality
        timing_analysis = self._analyze_signal_timing(
            current_date, signal, ml_confidence, composite_tech,
            ml_history, tech_history, price_history
        )
        
        # 4. Check for missed opportunities
        missed_signals = self._check_missed_opportunities(
            current_date, signal, ml_confidence, composite_tech,
            ml_history, tech_history, price_history.tail(10)
        )
        
        # 5. Overall trade recommendation
        should_trade = (filter_result['enter_trade'] and 
                       timing_analysis['timing_score'] >= 0.6 and
                       risk_levels['risk_reward_ratio'] >= 1.5)
        
        return {
            'should_enter_trade': should_trade,
            'signal_direction': signal,
            'confidence': ml_confidence,
            'composite_tech': composite_tech,
            'filter_result': filter_result,
            'timing_analysis': timing_analysis,
            'risk_levels': risk_levels,
            'missed_opportunities': missed_signals,
            'recommendation': self._generate_recommendation(
                should_trade, signal, timing_analysis, risk_levels, filter_result
            )
        }
    
    def _analyze_signal_timing(self,
                              current_date: pd.Timestamp,
                              signal: int,
                              ml_confidence: float,
                              composite_tech: float,
                              ml_history: pd.Series,
                              tech_history: pd.Series,
                              price_history: pd.DataFrame) -> Dict:
        """
        Analyze the timing quality of the current signal
        """
        
        if len(ml_history) < 10:
            return {'timing_score': 0.5, 'timing_reason': 'Insufficient history'}
        
        # Get recent data for analysis
        recent_ml = ml_history.tail(10)
        recent_tech = tech_history.tail(10)
        recent_prices = price_history.tail(10)
        
        timing_factors = []
        timing_reasons = []
        
        # 1. Momentum confirmation
        ml_momentum = self._calculate_momentum(recent_ml, signal)
        tech_momentum = self._calculate_momentum(recent_tech, signal)
        
        momentum_score = (ml_momentum + tech_momentum) / 2
        timing_factors.append(momentum_score)
        timing_reasons.append(f'Momentum: ML={ml_momentum:.2f}, Tech={tech_momentum:.2f}')
        
        # 2. Divergence analysis
        divergence_score = self._analyze_divergence(recent_ml, recent_tech, recent_prices, signal)
        timing_factors.append(divergence_score)
        timing_reasons.append(f'Divergence quality: {divergence_score:.2f}')
        
        # 3. Valley/Peak analysis (addresses user's specific concern)
        peak_valley_score = self._analyze_peaks_valleys(recent_ml, recent_tech, signal)
        timing_factors.append(peak_valley_score)
        timing_reasons.append(f'Peak/Valley timing: {peak_valley_score:.2f}')
        
        # 4. Stability check (avoid noisy signals)
        stability_score = self._check_signal_stability(recent_ml, recent_tech)
        timing_factors.append(stability_score)
        timing_reasons.append(f'Signal stability: {stability_score:.2f}')
        
        # Calculate overall timing score
        timing_score = np.mean(timing_factors)
        
        return {
            'timing_score': round(timing_score, 3),
            'timing_reason': ' | '.join(timing_reasons),
            'momentum_score': momentum_score,
            'divergence_score': divergence_score,
            'peak_valley_score': peak_valley_score,
            'stability_score': stability_score
        }
    
    def _calculate_momentum(self, series: pd.Series, signal_direction: int) -> float:
        """Calculate momentum alignment with signal direction"""
        if len(series) < 3:
            return 0.5
        
        # Calculate recent trend
        recent_slope = (series.iloc[-1] - series.iloc[-3]) / 2
        
        if signal_direction == 1:  # Long signal
            return max(0, min(1, 0.5 + recent_slope))
        else:  # Short signal
            return max(0, min(1, 0.5 - recent_slope))
    
    def _analyze_divergence(self, ml_series: pd.Series, tech_series: pd.Series, 
                           price_series: pd.DataFrame, signal_direction: int) -> float:
        """Analyze bullish/bearish divergence quality"""
        
        if len(ml_series) < 5:
            return 0.5
        
        # Calculate price momentum
        price_momentum = (price_series['close'].iloc[-1] - price_series['close'].iloc[-3]) / price_series['close'].iloc[-3]
        
        # Calculate indicator momentum
        ml_momentum = ml_series.iloc[-1] - ml_series.iloc[-3]
        tech_momentum = tech_series.iloc[-1] - tech_series.iloc[-3]
        
        if signal_direction == 1:  # Long signal - look for bullish divergence
            # Price declining but indicators rising = bullish divergence (good)
            if price_momentum < 0 and ml_momentum > 0 and tech_momentum > 0:
                return 0.9  # Strong bullish divergence
            elif price_momentum > 0 and ml_momentum > 0 and tech_momentum > 0:
                return 0.7  # Price and indicators both rising (good confirmation)
            else:
                return 0.3  # Poor divergence
        
        else:  # Short signal - look for bearish divergence
            # Price rising but indicators falling = bearish divergence (good)
            if price_momentum > 0 and ml_momentum < 0 and tech_momentum < 0:
                return 0.9  # Strong bearish divergence
            elif price_momentum < 0 and ml_momentum < 0 and tech_momentum < 0:
                return 0.7  # Price and indicators both falling (good confirmation)
            else:
                return 0.3  # Poor divergence
    
    def _analyze_peaks_valleys(self, ml_series: pd.Series, tech_series: pd.Series, 
                              signal_direction: int) -> float:
        """
        Analyze peak/valley timing to avoid entries at local extremes
        Addresses user's concern about July 14, Aug 11-14, Oct 5-6 timing issues
        """
        if len(ml_series) < 7:
            return 0.5
        
        # Find local peaks and valleys in recent data
        ml_recent = ml_series.tail(7)
        tech_recent = tech_series.tail(7)
        
        # Check if current values are at local extremes
        ml_current = ml_recent.iloc[-1]
        tech_current = tech_recent.iloc[-1]
        
        ml_rank_in_recent = (ml_recent <= ml_current).sum() / len(ml_recent)
        tech_rank_in_recent = (tech_recent <= tech_current).sum() / len(tech_recent)
        
        if signal_direction == 1:  # Long signal
            # For long signals, we want to avoid buying at local peaks
            # Good: ML confidence in valley (low rank), tech also not at peak
            if ml_rank_in_recent <= 0.3 and tech_rank_in_recent <= 0.7:
                return 0.9  # Excellent - buying near ML valley
            elif ml_rank_in_recent <= 0.5:
                return 0.7  # Good timing
            elif ml_rank_in_recent >= 0.8:
                return 0.2  # Poor - buying at local peak
            else:
                return 0.5  # Neutral
        
        else:  # Short signal
            # For short signals, we want to avoid selling at local valleys
            # Good: ML confidence at peak (high rank), tech also elevated
            if ml_rank_in_recent >= 0.7 and tech_rank_in_recent >= 0.3:
                return 0.9  # Excellent - selling near ML peak
            elif ml_rank_in_recent >= 0.5:
                return 0.7  # Good timing
            elif ml_rank_in_recent <= 0.2:
                return 0.2  # Poor - selling at local valley
            else:
                return 0.5  # Neutral
    
    def _check_signal_stability(self, ml_series: pd.Series, tech_series: pd.Series) -> float:
        """Check if signals are stable (not too noisy)"""
        if len(ml_series) < 5:
            return 0.5
        
        # Calculate volatility of recent signals
        ml_std = ml_series.tail(5).std()
        tech_std = tech_series.tail(5).std()
        
        # Lower volatility = more stable = better score
        ml_stability = max(0, 1 - ml_std / 0.5)  # Normalize by expected range
        tech_stability = max(0, 1 - tech_std / 0.5)
        
        return (ml_stability + tech_stability) / 2
    
    def _check_missed_opportunities(self, 
                                   current_date: pd.Timestamp,
                                   current_signal: int,
                                   ml_confidence: float,
                                   composite_tech: float,
                                   ml_history: pd.Series,
                                   tech_history: pd.Series,
                                   recent_prices: pd.DataFrame) -> List[Dict]:
        """
        Detect missed trading opportunities in recent history
        Addresses user's question about missing signals on Oct 27, Nov 11
        """
        missed_signals = []
        
        if len(ml_history) < 10:
            return missed_signals
        
        # Look at last 10 periods for missed opportunities
        for i in range(len(ml_history) - 10, len(ml_history) - 1):
            if i < 0:
                continue
            
            date = ml_history.index[i]
            ml_val = ml_history.iloc[i]
            tech_val = tech_history.iloc[i]
            
            # Check for strong signals that should have triggered trades
            # Sell signal opportunities
            if (abs(ml_val) > 0.4 and ml_val < -0.3 and 
                abs(tech_val) > 0.3 and tech_val > 0.2):  # ML bearish, tech at peak
                
                missed_signals.append({
                    'date': date,
                    'type': 'MISSED_SELL',
                    'ml_confidence': round(ml_val, 3),
                    'composite_tech': round(tech_val, 3),
                    'reason': 'Strong bearish ML with tech at peak - classic short setup'
                })
            
            # Buy signal opportunities
            elif (abs(ml_val) > 0.4 and ml_val > 0.3 and
                  abs(tech_val) > 0.3 and tech_val < -0.2):  # ML bullish, tech in valley
                
                missed_signals.append({
                    'date': date,
                    'type': 'MISSED_BUY', 
                    'ml_confidence': round(ml_val, 3),
                    'composite_tech': round(tech_val, 3),
                    'reason': 'Strong bullish ML with tech in valley - classic long setup'
                })
        
        return missed_signals
    
    def _generate_recommendation(self,
                               should_trade: bool,
                               signal: int,
                               timing_analysis: Dict,
                               risk_levels: Dict,
                               filter_result: Dict) -> str:
        """Generate human-readable trading recommendation"""
        
        if not should_trade:
            if not filter_result['enter_trade']:
                return f"❌ Skip Trade: {filter_result['reason']}"
            elif timing_analysis['timing_score'] < 0.6:
                return f"⏳ Wait for Better Timing: {timing_analysis['timing_reason']}"
            elif risk_levels['risk_reward_ratio'] < 1.5:
                return f"⚠️ Poor Risk/Reward: {risk_levels['risk_reward_ratio']:.2f}:1 ratio"
            else:
                return "❌ Skip Trade: Multiple factors unfavorable"
        
        signal_type = "BUY" if signal == 1 else "SELL"
        confidence_level = "High" if abs(risk_levels['confidence_multiplier']) > 1.3 else "Medium"
        
        return f"✅ {signal_type} Signal - {confidence_level} Confidence | " \
               f"SL: {risk_levels['stop_loss_pct']:.1f}% | " \
               f"TP: {risk_levels['take_profit_pct']:.1f}% | " \
               f"R/R: {risk_levels['risk_reward_ratio']:.1f}:1"


def analyze_historical_missed_signals(ml_confidence_history: pd.Series,
                                     composite_tech_history: pd.Series,
                                     price_history: pd.DataFrame,
                                     actual_signals: pd.Series) -> Dict:
    """
    Analyze historical periods to find missed trading opportunities
    Specifically addresses user's questions about Oct 27, Nov 11, etc.
    """
    engine = EnhancedTradingEngine()
    analysis_results = []
    
    # Look at each historical period
    for i in range(10, len(ml_confidence_history)):
        date = ml_confidence_history.index[i]
        ml_conf = ml_confidence_history.iloc[i]
        tech_val = composite_tech_history.iloc[i]
        actual_signal = actual_signals.iloc[i] if i < len(actual_signals) else 0
        
        # Get historical context
        ml_history = ml_confidence_history.iloc[:i]
        tech_history = composite_tech_history.iloc[:i]
        price_hist = price_history.iloc[:i]
        
        # Evaluate what the enhanced engine would have done
        opportunity = engine.evaluate_trading_opportunity(
            current_date=date,
            signal=1 if ml_conf > 0.3 else -1 if ml_conf < -0.3 else 0,
            ml_confidence=ml_conf,
            composite_tech=tech_val,
            price=price_history['close'].iloc[i],
            price_history=price_hist,
            ml_history=ml_history,
            tech_history=tech_history
        )
        
        # Check for missed opportunities
        if (opportunity['should_enter_trade'] and actual_signal == 0):
            analysis_results.append({
                'date': date,
                'type': 'MISSED_OPPORTUNITY',
                'recommended_signal': opportunity['signal_direction'],
                'ml_confidence': ml_conf,
                'composite_tech': tech_val,
                'timing_score': opportunity['timing_analysis']['timing_score'],
                'recommendation': opportunity['recommendation']
            })
        
        elif (not opportunity['should_enter_trade'] and actual_signal != 0):
            analysis_results.append({
                'date': date,
                'type': 'AVOIDED_POOR_TRADE',
                'actual_signal': actual_signal,
                'ml_confidence': ml_conf,
                'composite_tech': tech_val,
                'timing_score': opportunity['timing_analysis']['timing_score'],
                'reason': opportunity['recommendation']
            })
    
    return {
        'missed_opportunities': [r for r in analysis_results if r['type'] == 'MISSED_OPPORTUNITY'],
        'avoided_poor_trades': [r for r in analysis_results if r['type'] == 'AVOIDED_POOR_TRADE'],
        'total_analysis_points': len(analysis_results)
    }
