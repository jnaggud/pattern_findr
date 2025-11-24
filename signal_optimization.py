"""
Enhancement #1: Signal Optimization Module
Analyzes signal distribution and optimizes thresholds for better frequency vs quality balance
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

def analyze_signal_distribution(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Analyze the distribution of buy/sell signals to understand why we get so few trades
    """
    from optimization import universal_strategy
    
    # Get current signals
    signals = universal_strategy(data, params)
    
    # Simulate signal scores at different thresholds
    buy_scores, sell_scores = simulate_signal_scores(data, params)
    
    analysis = {
        'current_trades': (signals != 0).sum(),
        'current_buy_signals': (signals == 1).sum(),
        'current_sell_signals': (signals == -1).sum(),
        'signal_distribution': {
            'buy_score_max': buy_scores.max(),
            'buy_score_mean': buy_scores.mean(),
            'buy_score_std': buy_scores.std(),
            'sell_score_max': sell_scores.max(),
            'sell_score_mean': sell_scores.mean(),
            'sell_score_std': sell_scores.std(),
        },
        'threshold_analysis': analyze_threshold_impact(buy_scores, sell_scores, params)
    }
    
    return analysis

def simulate_signal_scores(data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate the signal scoring process to get raw scores before thresholds
    """
    # Get active indicators (copy logic from universal_strategy)
    meta_flags = {
        'buy_score_threshold', 'sell_score_threshold', 'signal_persistence_days',
        'min_hold_days', 'require_confirmation', 'confirmation_weight',
        'use_weighted_scoring', 'use_volatility_sizing', 'use_dynamic_thresholds',
        'use_regime_detection', 'use_dynamic_position_sizing', 'use_trend_filter',
        'trend_adx_threshold', 'trend_rsi_oversold', 'trend_willr_threshold',
        'trend_stoch_threshold', 'trend_sma_period', 'trend_filter_weight',
        'volatility_threshold', 'volatility_multiplier', 'volatility_percentile_threshold',
        'high_vol_threshold_reduction', 'buy_pct_min', 'buy_pct_max',
        'sell_pct_min', 'sell_pct_max', 'use_indicators_ready'
    }
    
    active_indicators = [
        param.replace('use_', '')
        for param in params.keys()
        if param.startswith('use_')
        and params[param] is True
        and param not in meta_flags
    ]
    
    # Initialize scores
    signal_persistence_days = params.get('signal_persistence_days', 2)
    buy_score_persistent = pd.Series(0.0, index=data.index)
    sell_score_persistent = pd.Series(0.0, index=data.index)
    
    # Process each indicator (simplified version of universal_strategy logic)
    for indicator in active_indicators:
        if indicator not in data.columns:
            continue
            
        if indicator.startswith('pattern_') or indicator.startswith('dl_signal_'):
            # Boolean indicators
            buy_param = params.get(f'{indicator}_buy')
            sell_param = params.get(f'{indicator}_sell')
            
            if buy_param is not None and data[indicator].any():
                raw_signals = (data[indicator] == buy_param).astype(float)
                buy_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
            if sell_param is not None and data[indicator].any():
                raw_signals = (data[indicator] == sell_param).astype(float)
                sell_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
        else:
            # Numerical indicators
            buy_threshold = params.get(f'{indicator}_buy')
            sell_threshold = params.get(f'{indicator}_sell')
            
            if buy_threshold is not None:
                raw_signals = (data[indicator] < buy_threshold).astype(float)
                buy_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
            if sell_threshold is not None:
                raw_signals = (data[indicator] > sell_threshold).astype(float)
                sell_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
    
    return buy_score_persistent, sell_score_persistent

def analyze_threshold_impact(buy_scores: pd.Series, sell_scores: pd.Series, params: Dict) -> Dict:
    """
    Analyze how different thresholds would affect signal frequency
    """
    current_buy_thresh = params['buy_score_threshold']
    current_sell_thresh = params['sell_score_threshold']
    
    # Test different thresholds
    threshold_tests = {}
    
    # Test buy thresholds from 1 to 15
    for buy_thresh in range(1, 16):
        # Test sell thresholds from 1 to 15
        for sell_thresh in range(1, 16):
            buy_signals = buy_scores >= buy_thresh
            sell_signals = sell_scores >= sell_thresh
            
            total_signals = buy_signals.sum() + sell_signals.sum()
            signal_ratio = buy_signals.sum() / max(1, sell_signals.sum())
            
            threshold_tests[f"buy_{buy_thresh}_sell_{sell_thresh}"] = {
                'buy_signals': int(buy_signals.sum()),
                'sell_signals': int(sell_signals.sum()),
                'total_signals': int(total_signals),
                'buy_sell_ratio': round(signal_ratio, 2),
                'improvement_vs_current': int(total_signals) - int((buy_scores >= current_buy_thresh).sum() + (sell_scores >= current_sell_thresh).sum())
            }
    
    return threshold_tests

def suggest_optimal_thresholds(analysis: Dict, target_trade_count: int = 50) -> Dict:
    """
    Suggest optimal thresholds based on analysis to achieve target trade count
    """
    threshold_analysis = analysis['threshold_analysis']
    
    # Find thresholds closest to target trade count
    best_options = []
    
    for config_name, metrics in threshold_analysis.items():
        trade_count = metrics['total_signals']
        buy_sell_ratio = metrics['buy_sell_ratio']
        
        # Score based on closeness to target and reasonable buy/sell ratio
        distance_score = abs(trade_count - target_trade_count)
        ratio_score = abs(buy_sell_ratio - 1.0)  # Prefer balanced buy/sell
        
        combined_score = distance_score + (ratio_score * 10)  # Weight ratio balance
        
        best_options.append({
            'config': config_name,
            'trade_count': trade_count,
            'buy_sell_ratio': buy_sell_ratio,
            'score': combined_score,
            'metrics': metrics
        })
    
    # Sort by combined score and return top 5 options
    best_options.sort(key=lambda x: x['score'])
    
    return {
        'current_performance': {
            'trade_count': analysis['current_trades'],
            'buy_signals': analysis['current_buy_signals'], 
            'sell_signals': analysis['current_sell_signals']
        },
        'recommended_thresholds': best_options[:5],
        'signal_distribution_insights': analysis['signal_distribution']
    }

def enhance_signal_scoring(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Enhancement #1: Implement improved signal scoring with confidence weighting
    """
    # Get raw signal scores
    buy_scores, sell_scores = simulate_signal_scores(data, params)
    
    # Calculate signal confidence based on score distribution
    buy_confidence = calculate_signal_confidence(buy_scores)
    sell_confidence = calculate_signal_confidence(sell_scores)
    
    # Adaptive thresholds based on market volatility
    volatility = calculate_market_volatility(data)
    adaptive_buy_thresh, adaptive_sell_thresh = calculate_adaptive_thresholds(
        params, volatility, buy_scores, sell_scores
    )
    
    enhanced_params = params.copy()
    enhanced_params.update({
        'adaptive_buy_threshold': adaptive_buy_thresh,
        'adaptive_sell_threshold': adaptive_sell_thresh,
        'buy_confidence_weights': buy_confidence,
        'sell_confidence_weights': sell_confidence,
        'market_volatility': volatility
    })
    
    return enhanced_params

def calculate_signal_confidence(scores: pd.Series) -> pd.Series:
    """
    Calculate confidence weights based on signal strength relative to historical distribution
    """
    # Use rolling percentile to calculate confidence
    rolling_window = min(50, len(scores) // 4)  # 50-day or 25% of data
    
    # Calculate rolling percentiles for confidence scoring
    rolling_75th = scores.rolling(window=rolling_window, min_periods=10).quantile(0.75)
    rolling_90th = scores.rolling(window=rolling_window, min_periods=10).quantile(0.90)
    
    # Confidence: 0.5 (normal) to 2.0 (extremely high confidence)
    confidence = pd.Series(0.5, index=scores.index)
    confidence.loc[scores >= rolling_75th] = 1.0  # Above 75th percentile
    confidence.loc[scores >= rolling_90th] = 2.0  # Above 90th percentile (rare, high confidence)
    
    return confidence.fillna(0.5)

def calculate_market_volatility(data: pd.DataFrame) -> pd.Series:
    """
    Calculate rolling market volatility for adaptive thresholds
    """
    if 'close' in data.columns:
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)  # Annualized
        return volatility.fillna(volatility.mean())
    else:
        return pd.Series(0.2, index=data.index)  # Default moderate volatility

def calculate_adaptive_thresholds(params: Dict, volatility: pd.Series, 
                                buy_scores: pd.Series, sell_scores: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate adaptive thresholds based on market conditions
    """
    base_buy_thresh = params['buy_score_threshold']
    base_sell_thresh = params['sell_score_threshold']
    
    # In high volatility, lower thresholds to catch more opportunities
    # In low volatility, raise thresholds to be more selective
    volatility_median = volatility.median()
    
    volatility_multiplier = pd.Series(1.0, index=volatility.index)
    
    # High volatility (above median): reduce thresholds by up to 30%
    high_vol_mask = volatility > volatility_median * 1.2
    volatility_multiplier.loc[high_vol_mask] = 0.7
    
    # Low volatility (below median): increase thresholds by up to 20%
    low_vol_mask = volatility < volatility_median * 0.8
    volatility_multiplier.loc[low_vol_mask] = 1.2
    
    adaptive_buy_thresh = (base_buy_thresh * volatility_multiplier).round().astype(int)
    adaptive_sell_thresh = (base_sell_thresh * volatility_multiplier).round().astype(int)
    
    # Ensure minimum thresholds
    adaptive_buy_thresh = adaptive_buy_thresh.clip(lower=1)
    adaptive_sell_thresh = adaptive_sell_thresh.clip(lower=1)
    
    return adaptive_buy_thresh, adaptive_sell_thresh

def optimize_signal_parameters(data: pd.DataFrame, params: Dict, target_trades: int = 40) -> Dict:
    """
    Main function to optimize signal parameters for Enhancement #1
    """
    print("🔍 Analyzing current signal distribution...")
    analysis = analyze_signal_distribution(data, params)
    
    print("📊 Finding optimal thresholds...")
    optimization_results = suggest_optimal_thresholds(analysis, target_trades)
    
    print("🚀 Enhancing signal scoring...")
    enhanced_params = enhance_signal_scoring(data, params)
    
    return {
        'analysis': analysis,
        'optimization_results': optimization_results,
        'enhanced_params': enhanced_params,
        'recommendations': {
            'summary': f"Current: {analysis['current_trades']} trades. Target: {target_trades} trades.",
            'best_threshold_config': optimization_results['recommended_thresholds'][0] if optimization_results['recommended_thresholds'] else None,
            'signal_insights': optimization_results['signal_distribution_insights']
        }
    }
