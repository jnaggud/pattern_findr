"""
Parameter Optimization System for Peak/Valley Detection
Finds optimal parameter combinations through systematic backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class ParameterOptimizer:
    """Systematic parameter search for peak/valley detection and ML training"""
    
    def __init__(self):
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
    def define_search_space(self) -> Dict:
        """Define parameter search space"""
        return {
            'window_size': [2, 3, 4, 5, 7, 10],
            'lead_time_days': [0, 1, 2, 3],
            'min_peak_height_pct': [1.0, 2.0, 3.0, 4.0, 5.0],
            'class_weight_ratio': [1, 3, 5, 7, 10, 15],
            'detection_method': ['predictive'],  # Fixed method name for display
            'model_type': ['xgboost', 'lightgbm', 'random_forest']
        }
    
    def evaluate_params(self, params: Dict, data: pd.DataFrame,
                       ml_features: pd.DataFrame,
                       train_end_date: str, test_start_date: str) -> Dict:
        """Evaluate a single parameter combination"""
        try:
            from ml_models import TradingMLModels
            from scipy.signal import argrelextrema
            
            # 1. Generate predictive labels using the same approach as peak_valley_ml_page.py
            window = params['window_size']
            lead_time_days = params['lead_time_days']
            min_peak_height_pct = params['min_peak_height_pct']
            
            # Find historical peaks and valleys
            highs = data['high'].values
            lows = data['low'].values
            
            # Use window parameter for detection
            peak_indices = argrelextrema(highs, np.greater, order=window)[0]
            valley_indices = argrelextrema(lows, np.less, order=window)[0]
            
            # Filter peaks/valleys by minimum height requirement
            filtered_peak_indices = []
            for idx in peak_indices:
                if idx > 0 and idx < len(highs) - 1:
                    peak_height = highs[idx]
                    left_val = highs[max(0, idx - window)]
                    right_val = highs[min(len(highs) - 1, idx + window)]
                    min_surrounding = min(left_val, right_val)
                    if (peak_height - min_surrounding) / min_surrounding * 100 >= min_peak_height_pct:
                        filtered_peak_indices.append(idx)
            
            filtered_valley_indices = []
            for idx in valley_indices:
                if idx > 0 and idx < len(lows) - 1:
                    valley_depth = lows[idx]
                    left_val = lows[max(0, idx - window)]
                    right_val = lows[min(len(lows) - 1, idx + window)]
                    max_surrounding = max(left_val, right_val)
                    if (max_surrounding - valley_depth) / max_surrounding * 100 >= min_peak_height_pct:
                        filtered_valley_indices.append(idx)
            
            # CREATE PREDICTIVE LABELS (lead_time_days before peak/valley)
            labels = pd.Series(0, index=data.index, name='signal')  # Default HOLD
            
            # Label lead_time_days BEFORE peaks as SELL (-1)
            for peak_idx in filtered_peak_indices:
                if peak_idx >= lead_time_days:
                    prev_day = data.index[peak_idx - lead_time_days]
                    labels.loc[prev_day] = -1
            
            # Label lead_time_days BEFORE valleys as BUY (1)
            for valley_idx in filtered_valley_indices:
                if valley_idx >= lead_time_days:
                    prev_day = data.index[valley_idx - lead_time_days]
                    labels.loc[prev_day] = 1

            # 2. Use precomputed ML features (compute once per optimization run)
            # Handle NaN values defensively
            ml_features = ml_features.fillna(0)
            
            # Align features and labels
            common_index = ml_features.index.intersection(labels.index)
            ml_features = ml_features.loc[common_index]
            labels = labels.loc[common_index]
            
            # 3. Split data
            # Convert string dates to timezone-aware timestamps if needed
            if hasattr(ml_features.index, 'tz') and ml_features.index.tz is not None:
                train_end_ts = pd.Timestamp(train_end_date).tz_localize(ml_features.index.tz)
                test_start_ts = pd.Timestamp(test_start_date).tz_localize(ml_features.index.tz)
            else:
                train_end_ts = pd.Timestamp(train_end_date)
                test_start_ts = pd.Timestamp(test_start_date)
            
            train_mask = ml_features.index <= train_end_ts
            test_mask = ml_features.index >= test_start_ts
            
            X_train = ml_features[train_mask]
            y_train = labels[train_mask]
            X_test = ml_features[test_mask]
            y_test = labels[test_mask]
            
            # Skip if insufficient data
            if len(X_train) < 50 or len(X_test) < 20:
                return {'error': 'Insufficient data', 'params': params}
            
            # 4. Train model (NO nested Optuna here; keep each combo fast)
            ml_models = TradingMLModels()
            
            # Set class weights
            class_weights = {
                -1: params['class_weight_ratio'],  # SELL
                0: 1,                              # HOLD
                1: params['class_weight_ratio']    # BUY
            }
            ml_models.class_weights = class_weights

            # Cap per-model parallelism so outer optimizer parallelism can saturate CPU.
            model_params = {}
            if params['model_type'] == 'random_forest':
                # RF defaults to n_jobs=-1 in this codebase; override to avoid oversubscription
                model_params['n_jobs'] = 1
                model_params['class_weight'] = class_weights
            elif params['model_type'] == 'xgboost':
                model_params['n_jobs'] = 1
            elif params['model_type'] == 'lightgbm':
                # LightGBM uses num_threads internally
                model_params['num_threads'] = 1

            train_result = ml_models.train_model(
                params['model_type'],
                X_train,
                y_train,
                model_params=model_params,
                use_scaling=True
            )
            
            if 'error' in train_result:
                return {'error': train_result['error'], 'params': params}
            
            # 5. Generate test predictions
            predictions, probabilities = ml_models.predict_signals(X_test, params['model_type'])
            
            # 6. Backtest on test data (align OHLC with feature split)
            test_data = data.loc[X_test.index]
            backtest_results = self._run_backtest(test_data, predictions)
            
            # 7. Calculate metrics
            metrics = {
                'params': params,
                'train_score': train_result.get('train_accuracy', 0),
                'cv_score': train_result.get('cv_mean', 0),
                'test_trades': backtest_results['num_trades'],
                'test_return': backtest_results['total_return'],
                'test_win_rate': backtest_results['win_rate'],
                'test_sharpe': backtest_results['sharpe_ratio'],
                'test_max_drawdown': backtest_results['max_drawdown'],
                'signal_distribution': {
                    'buy': int((predictions == 1).sum()),
                    'sell': int((predictions == -1).sum()),
                    'hold': int((predictions == 0).sum())
                },
                'optimization': train_result.get('optimization', {})
            }
            
            # Composite score for ranking
            metrics['composite_score'] = self._calculate_composite_score(metrics)
            
            return metrics
            
        except Exception as e:
            import traceback
            return {
                'error': str(e), 
                'traceback': traceback.format_exc(),
                'params': params
            }
    
    def _apply_lead_time_old(self, labels_raw: pd.Series, data: pd.DataFrame, 
                        lead_days: int) -> pd.Series:
        """Apply lead time to labels for predictive signals"""
        if lead_days == 0:
            return labels_raw
        
        # Find peaks and valleys
        peak_indices = np.where(labels_raw == -1)[0]  # Peaks -> SELL
        valley_indices = np.where(labels_raw == 1)[0]  # Valleys -> BUY
        
        # Create new labels
        labels = pd.Series(0, index=labels_raw.index)  # Default HOLD
        
        # Apply lead time
        for peak_idx in peak_indices:
            if peak_idx >= lead_days:
                labels.iloc[peak_idx - lead_days] = -1  # SELL signal before peak
        
        for valley_idx in valley_indices:
            if valley_idx >= lead_days:
                labels.iloc[valley_idx - lead_days] = 1  # BUY signal before valley
        
        return labels
    
    def _run_backtest(self, data: pd.DataFrame, signals: np.ndarray) -> Dict:
        """Run simple backtest to evaluate signals"""
        capital = 100000
        position = None
        trades = []
        
        min_len = min(len(data), len(signals))
        
        for i in range(min_len):
            price = data['close'].iloc[i]
            signal = signals[i]
            
            if signal == 1 and position is None:  # BUY
                position = {'entry': price, 'shares': capital / price}
                
            elif signal == -1 and position is not None:  # SELL
                exit_value = position['shares'] * price
                profit = exit_value - capital
                trades.append({
                    'profit': profit,
                    'return': profit / capital
                })
                position = None
                capital = exit_value
        
        # Close final position
        if position:
            final_price = data['close'].iloc[-1]
            exit_value = position['shares'] * final_price
            profit = exit_value - capital
            trades.append({
                'profit': profit,
                'return': profit / capital
            })
            capital = exit_value
        
        # Calculate metrics
        if not trades:
            return {
                'num_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        returns = [t['return'] for t in trades]
        winning_trades = sum(1 for r in returns if r > 0)
        
        # Calculate Sharpe ratio (annualized)
        if len(returns) > 1:
            daily_returns = np.array(returns)
            sharpe = np.sqrt(252) * (np.mean(daily_returns) / (np.std(daily_returns) + 1e-6))
        else:
            sharpe = 0
        
        return {
            'num_trades': len(trades),
            'total_return': (capital - 100000) / 100000,
            'win_rate': winning_trades / len(trades) if trades else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': min(returns) if returns else 0
        }
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite score for ranking parameter combinations"""
        # Skip if error or no trades
        if 'error' in metrics or metrics['test_trades'] == 0:
            return -1000
        
        # Weighted scoring
        score = 0
        score += metrics['test_return'] * 100  # Return weight
        score += metrics['test_win_rate'] * 50  # Win rate weight
        score += metrics['test_sharpe'] * 20  # Sharpe weight
        score -= abs(metrics['test_max_drawdown']) * 30  # Drawdown penalty
        score += min(metrics['test_trades'], 10) * 2  # Trade frequency bonus (capped)
        
        # Penalty for overfitting (train vs CV gap)
        overfit_penalty = abs(metrics['train_score'] - metrics['cv_score']) * 50
        score -= overfit_penalty
        
        return score
    
    def optimize_parameters(self, data: pd.DataFrame, train_end_date: str, 
                           test_start_date: str, max_workers: int = None) -> Dict:
        """Run full parameter optimization"""
        print(f"🚀 Starting parameter optimization at {datetime.now()}")

        # Precompute ML features ONCE to avoid repeated indicator generation, DL model training, etc.
        # DL features are explicitly disabled for optimization speed/stability.
        from ml_feature_engineer import MLFeatureEngineer
        engineer = MLFeatureEngineer()
        ml_features = engineer.prepare_ml_dataset(
            data,
            include_lagged=True,
            include_rolling=True,
            feature_selection=True,
            use_dl_features=False
        )
        ml_features = ml_features.fillna(0)
        
        # Get search space
        search_space = self.define_search_space()
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(all_combinations)
        print(f"📊 Testing {total_combinations} parameter combinations")
        print(f"⏱️  Estimated time: {total_combinations * 30 / 60:.1f} - {total_combinations * 60 / 60:.1f} minutes")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(os.cpu_count() - 1, 8)  # Leave one CPU free
        
        print(f"💪 Using {max_workers} parallel workers")
        
        # Process combinations in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {}
            for combination in all_combinations:
                params = dict(zip(param_names, combination))
                future = executor.submit(
                    self.evaluate_params,
                    params,
                    data,
                    ml_features,
                    train_end_date,
                    test_start_date
                )
                future_to_params[future] = params
            
            # Process completed tasks
            for future in as_completed(future_to_params):
                completed += 1
                result = future.result()
                self.results.append(result)
                
                # Update best if no error and better score
                if 'error' not in result:
                    score = result['composite_score']
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = result['params']
                        print(f"\n✨ New best! Score: {score:.2f}")
                        print(f"   Params: {self.best_params}")
                        print(f"   Return: {result['test_return']:.2%}, Sharpe: {result['test_sharpe']:.2f}")
                
                # Progress update
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
        
        # Sort results by score
        valid_results = [r for r in self.results if 'error' not in r]
        valid_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"parameter_optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'top_10_results': valid_results[:10] if valid_results else [],
                'summary': {
                    'total_tested': total_combinations,
                    'successful': len(valid_results),
                    'failed': len(self.results) - len(valid_results)
                }
            }, f, indent=2, default=str)
        
        print(f"\n🎯 Optimization complete!")
        print(f"📁 Results saved to: {results_file}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'top_results': valid_results[:10] if valid_results else [],
            'results_file': results_file
        }
