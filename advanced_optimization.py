"""
Advanced Optuna optimization with multiple parallelization strategies.
"""

import optuna
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from joblib import Parallel, delayed
import sqlite3
import tempfile
import os
from optimization import objective

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_optimization_distributed(data, n_trials=5000, n_jobs=None, method='multiprocessing'):
    """
    Advanced optimization with multiple parallelization strategies.
    
    Args:
        data: Market data for optimization
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs (None = auto-detect)
        method: 'multiprocessing', 'joblib', 'threading', or 'distributed'
    
    Returns:
        Top 3 trials
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    print(f"\n{'='*40}")
    print(f"ADVANCED OPTIMIZATION")
    print(f"Method: {method.upper()}")
    print(f"Trials: {n_trials}")
    print(f"Workers: {n_jobs}")
    print(f"{'='*40}\n")
    
    if method == 'multiprocessing':
        return _run_multiprocessing_optimization(data, n_trials, n_jobs)
    elif method == 'joblib':
        return _run_joblib_optimization(data, n_trials, n_jobs)
    elif method == 'threading':
        return _run_threading_optimization(data, n_trials, n_jobs)
    elif method == 'distributed':
        return _run_distributed_optimization(data, n_trials, n_jobs)
    else:
        raise ValueError(f"Unknown method: {method}")

def _run_multiprocessing_optimization(data, n_trials, n_jobs):
    """Standard multiprocessing approach."""
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, data),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    return _get_top_trials(study)

def _run_joblib_optimization(data, n_trials, n_jobs):
    """Joblib-based parallelization (often faster)."""
    study = optuna.create_study(direction='maximize')
    
    # Use joblib for parallel execution
    def run_single_trial(_):
        trial = study.ask()
        value = objective(trial, data)
        study.tell(trial, value)
        return value
    
    # Execute trials in parallel
    Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_single_trial)(i) for i in range(n_trials)
    )
    
    return _get_top_trials(study)

def _run_threading_optimization(data, n_trials, n_jobs):
    """Threading approach (good for I/O bound tasks)."""
    study = optuna.create_study(direction='maximize')
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        
        for _ in range(n_trials):
            trial = study.ask()
            future = executor.submit(objective, trial, data)
            futures.append((trial, future))
        
        # Collect results
        for trial, future in futures:
            try:
                value = future.result()
                study.tell(trial, value)
            except Exception as e:
                study.tell(trial, -1e9)
    
    return _get_top_trials(study)

def _run_distributed_optimization(data, n_trials, n_jobs):
    """
    Distributed optimization using shared database.
    This allows multiple processes to share the same study.
    """
    # Create temporary database for study
    db_path = tempfile.mktemp(suffix='.db')
    storage = f"sqlite:///{db_path}"
    
    try:
        study = optuna.create_study(
            direction='maximize',
            storage=storage,
            study_name='pattern_optimization'
        )
        
        # Run optimization with shared storage
        study.optimize(
            lambda trial: objective(trial, data),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        return _get_top_trials(study)
        
    finally:
        # Clean up temporary database
        if os.path.exists(db_path):
            os.remove(db_path)

def _get_top_trials(study):
    """Extract top 3 trials from study."""
    successful_trials = [t for t in study.trials if t.value is not None and t.value > -1e9]
    top_trials = sorted(successful_trials, key=lambda t: t.value, reverse=True)[:3]
    
    print(f"\n{'='*40}")
    print("TOP 3 STRATEGIES FOUND:")
    for i, trial in enumerate(top_trials, 1):
        active_indicators = [param.replace('use_', '') for param in trial.params.keys() 
                           if param.startswith('use_') and trial.params[param] == True]
        print(f"Rank {i}: {trial.value:.2f}% return - {len(active_indicators)} indicators")
        print(f"Trial #{trial.number}")
    print(f"{'='*40}\n")
    
    return top_trials

def benchmark_optimization_methods(data, n_trials=100):
    """
    Benchmark different optimization methods to find the fastest.
    
    Args:
        data: Market data
        n_trials: Number of trials for benchmarking
    
    Returns:
        dict: Performance results for each method
    """
    import time
    
    methods = ['multiprocessing', 'joblib', 'threading']
    results = {}
    
    print(f"\n🏁 BENCHMARKING OPTIMIZATION METHODS")
    print(f"Running {n_trials} trials per method...\n")
    
    for method in methods:
        print(f"Testing {method}...")
        start_time = time.time()
        
        try:
            top_trials = run_optimization_distributed(
                data, n_trials=n_trials, method=method
            )
            
            end_time = time.time()
            duration = end_time - start_time
            best_return = top_trials[0].value if top_trials else 0
            
            results[method] = {
                'duration': duration,
                'trials_per_second': n_trials / duration,
                'best_return': best_return,
                'success': True
            }
            
            print(f"✅ {method}: {duration:.1f}s ({n_trials/duration:.1f} trials/sec)")
            
        except Exception as e:
            results[method] = {
                'duration': float('inf'),
                'trials_per_second': 0,
                'best_return': 0,
                'success': False,
                'error': str(e)
            }
            print(f"❌ {method}: Failed - {e}")
    
    # Find best method
    best_method = min(
        [m for m in results if results[m]['success']], 
        key=lambda m: results[m]['duration']
    )
    
    print(f"\n🏆 BEST METHOD: {best_method.upper()}")
    print(f"Duration: {results[best_method]['duration']:.1f}s")
    print(f"Speed: {results[best_method]['trials_per_second']:.1f} trials/sec")
    
    return results, best_method

def get_recommended_settings():
    """Get recommended optimization settings based on system."""
    cpu_count = mp.cpu_count()
    
    if cpu_count >= 8:
        return {
            'n_jobs': cpu_count - 2,
            'method': 'joblib',
            'batch_size': 100
        }
    elif cpu_count >= 4:
        return {
            'n_jobs': cpu_count - 1,
            'method': 'multiprocessing',
            'batch_size': 50
        }
    else:
        return {
            'n_jobs': cpu_count,
            'method': 'threading',
            'batch_size': 25
        }

if __name__ == "__main__":
    # Test the advanced optimization
    print("Advanced Optimization Module Loaded")
    print(f"System has {mp.cpu_count()} CPU cores")
    
    settings = get_recommended_settings()
    print(f"Recommended settings: {settings}")
