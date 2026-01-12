# Pattern FindR - Comprehensive Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Parameter Optimization Workflow](#parameter-optimization-workflow)
4. [Recent Enhancements](#recent-enhancements)
5. [Market Data Database](#market-data-database)
6. [Price Range Prediction Suite](#price-range-prediction-suite)
7. [Installation & Setup](#installation--setup)
8. [Usage Guide](#usage-guide)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)
11. [Future Development](#future-development)

## System Overview

Pattern FindR is an advanced quantitative trading system that combines machine learning with technical analysis to identify profitable trading opportunities. The system is built with a modular architecture that allows for flexible strategy development and optimization.

### Key Features
- **Machine Learning-Powered Trading Signals**
- **Comprehensive Parameter Optimization**
- **Market Regime Detection**
- **Advanced Risk Management**
- **Interactive Web Interface**

## Core Components

### 1. Main Application (`peak_valley_ml_page.py`)
The primary interface built with Streamlit that provides:
- Data loading and preprocessing
- Interactive visualization
- Model training and evaluation
- Parameter optimization interface

### 2. Peak/Valley Detection (`peak_valley_detector.py`)
- Identifies market turning points
- Configurable window sizes and thresholds
- Supports multiple detection methods

### 3. Parameter Optimization (`parameter_optimizer.py`)
- Implements Optuna for hyperparameter optimization
- Supports both single and multi-objective optimization
- Includes early stopping and pruning

### 4. Feature Engineering (`ml_feature_engineer.py`)
- Generates technical indicators
- Creates lagged and rolling features
- Handles feature scaling and normalization

## Parameter Optimization Workflow

### Optimization Process
1. **Setup**: Define search space and optimization objectives
2. **Trial Execution**:
   - Sample parameters from search space
   - Train model with sampled parameters
   - Evaluate performance on validation set
   - Report metrics back to Optuna
3. **Result Analysis**:
   - Identify best performing parameters
   - Visualize optimization history
   - Save results for future use

### Key Optimization Parameters
| Parameter | Description | Range/Options |
|-----------|-------------|---------------|
| `window_size` | Lookback window for pattern detection | 3-20 days |
| `lead_time` | Days ahead to predict | 1-5 days |
| `min_peak_height` | Minimum price movement to consider | 0.01-0.10 |
| `class_weight_ratio` | Class imbalance handling | 1-10 |
| `indicator_thresholds` | Buy/sell signal thresholds | Indicator-specific |

### Optimization Results
Results are saved in JSON format with the following structure:
```json
{
  "best_params": {
    "window_size": 5,
    "lead_time": 2,
    "min_peak_height": 0.03,
    "class_weight_ratio": 5
  },
  "performance_metrics": {
    "sharpe_ratio": 2.5,
    "max_drawdown": -0.12,
    "win_rate": 0.68
  },
  "optimization_date": "2024-03-15"
}
```

## Recent Enhancements

### 1. Parameter Optimization Integration
- Added "Apply Now" button to immediately use optimized parameters
- Automatic clearing of cached data when parameters change
- Persistent storage of optimal settings using session state

### 2. Performance Improvements
- Precomputation of features to avoid redundant calculations
- Disabled deep learning features during optimization to reduce overhead
- Parallelized optimization trials for faster convergence

### 3. User Experience
- Progress indicators for long-running operations
- Clear error messages and validation
- Visual feedback when parameters are applied

## Market Data Database

The system uses a SQLite database (`market_data.db`) to cache market data locally. This eliminates yfinance rate limiting issues and enables fast parallel processing.

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MARKET DATA FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Historical Data (before today)                                 │
│   ─────────────────────────────                                  │
│   └─> SQLite Database (instant, no API calls)                    │
│                                                                   │
│   Today's Data (live trading)                                    │
│   ──────────────────────────                                     │
│   └─> Fresh yfinance fetch (ensures accuracy)                    │
│   └─> Saved to DB after market close                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Database Contents

| Symbol | Description | Usage |
|--------|-------------|-------|
| `^VIX` | VIX Volatility Index | Fear gauge, volatility features |
| `^VVIX` | VVIX (VIX of VIX) | Vol-of-vol, expecting VIX moves |
| `SPY` | S&P 500 ETF | Main ticker price data |
| `QQQ` | Nasdaq 100 ETF | SPY/QQQ correlation features |
| `^TNX` | 10Y Treasury Yield | Interest rate features |
| `^TYX` | 30Y Treasury Yield | Yield curve features |
| `HYG` | High Yield Bond ETF | Credit spread (risk-on) |
| `LQD` | Investment Grade Bond ETF | Credit spread (risk-off) |
| `DX-Y.NYB` | Dollar Index (DXY) | Currency strength features |
| `XLK, XLF, XLE...` | Sector ETFs (11 total) | Sector breadth analysis |

### Database Statistics

- **File:** `market_data.db` (~5-6 MB with 10 years of data)
- **Rows:** ~47,000+ price data rows
- **Date Range:** 10 years of historical data
- **Symbols:** 19 tracked instruments

### CLI Commands

```bash
# Sync all data (initial setup or daily update)
python market_data_db.py --sync

# Sync with custom date range
python market_data_db.py --sync --start 2016-01-01 --end 2026-01-11

# Show sync status for all symbols
python market_data_db.py --status

# Show database statistics
python market_data_db.py --stats

# Clear cache for specific symbol
python market_data_db.py --clear SPY

# Clear all cached data
python market_data_db.py --clear all
```

### Python API

```python
from market_data_db import get_market_db

# Get singleton database instance
db = get_market_db()

# Fetch VIX data (uses cache + fresh for today)
vix = db.get_vix('2024-01-01', '2026-01-11')

# Fetch sector ETF data
sectors = db.get_sectors('2024-01-01', '2026-01-11')

# Fetch credit spread data (HYG/LQD)
credit = db.get_credit_spreads('2024-01-01', '2026-01-11')

# Sync all data
db.sync_all(start_date='2016-01-01', verbose=True)

# Check database stats
stats = db.get_database_stats()
print(f"Rows: {stats['price_data_rows']}, Size: {stats['file_size_mb']} MB")
```

### Live Trading Data Flow

1. **App Startup:** Database auto-syncs missing historical data
2. **During Session:** Historical data from SQLite (instant)
3. **Today's Data:** Always fetched fresh from yfinance
4. **After Market Close:** Today's data saved to database
5. **Next Day:** Yesterday's data now served from cache

### Benefits for Parallel Processing

- **No Rate Limiting:** Each parallel worker reads from SQLite
- **Fast Feature Creation:** Data loads in <0.1s vs 10s+ from API
- **Offline Capable:** Historical analysis works without internet
- **Consistent Data:** All workers see identical historical data

---

## Price Range Prediction Suite

The Price Range Prediction Suite predicts tomorrow's high/low prices with confidence intervals. Uses Ridge regression with top-N correlation feature selection for optimal performance.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 PRICE RANGE PREDICTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: OHLCV Data + Market Features (VIX, Sectors, etc.)       │
│         ↓                                                        │
│  Feature Engineering: 200+ features                              │
│  - Technical indicators (RSI, MACD, ATR, Bollinger)             │
│  - Volatility features (VIX, VVIX, IV rank)                     │
│  - Market breadth (sector above SMA %)                          │
│  - Credit/Dollar/Treasury features                               │
│  - Super-features (composite indicators)                         │
│         ↓                                                        │
│  Feature Selection: Top 15 by correlation with target           │
│         ↓                                                        │
│  Ridge Regression with Parallel Alpha Search                     │
│  - Alpha grid: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50, 100]
│  - Cross-validation for optimal alpha                            │
│         ↓                                                        │
│  Output: Predicted High, Low, Range + Confidence Bands          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Model Configuration

Located in `price_prediction.py`:

```python
MODEL_CONFIG = {
    'model_type': 'ridge',        # Ridge regression (R²=0.56 vs XGBoost R²=0.33)
    'top_n_features': 15,         # Optimal feature count from testing
    'ridge_alpha': 1.0,           # Starting regularization strength
    'use_correlation_selection': True,  # Select features by correlation
}
```

### Why Ridge Over XGBoost?

| Metric | Ridge (Top 15) | XGBoost (All Features) |
|--------|----------------|------------------------|
| Test R² | 0.56 | 0.33 |
| Training Time | ~0.1s | ~5 min |
| Handles Collinearity | Yes (regularization) | Poorly (feature masking) |
| Parallel Friendly | Yes | Limited |

### Walk-Forward Analysis

Robust out-of-sample testing with daily model retraining.

**Settings in Streamlit UI:**

| Setting | Recommended | Description |
|---------|-------------|-------------|
| Test Period | 60 days | Number of days to test |
| Training Days | 500 days | Rolling window for training |
| Retrain Frequency | Every Day | How often to retrain |
| Optuna Trials | 100-300 | Not used for Ridge (has own alpha search) |
| Parallel Workers | CPU cores | For parallel walk-forward |
| Confidence Level | 68% | 1 std dev for trading bands |

### Parallel Walk-Forward Mode

When "Enable Parallel Walk-Forward" is checked:

1. **Requirement:** Retrain Frequency must be "Every Day"
2. **How it works:** Each day's model training runs in parallel
3. **Speed:** 60 days with 32 workers ≈ 2-4 minutes (vs 15-20 min sequential)
4. **Memory:** Higher usage (one model per worker)

```python
# Internally calls:
from price_prediction import run_parallel_walk_forward

results = run_parallel_walk_forward(
    pred_df=data,
    test_start_idx=start_idx,
    train_window=500,
    n_trials=100,
    n_workers=32,
    ci_level=0.68,
    polygon_api_key=api_key
)
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| High Containment | % of actual highs within predicted CI | >80% |
| Low Containment | % of actual lows within predicted CI | >80% |
| Full Containment | % where both high AND low in CI | >75% |
| MAE | Mean absolute error in dollars | <$3-4 |
| R² | Coefficient of determination | >0 (positive) |

### Output Files

Walk-forward analysis generates:

```
walk_forward/
├── SPY_wf_20251015_to_20260109_20260111_114931.csv      # Daily predictions
└── SPY_wf_20251015_to_20260109_20260111_114931_summary.txt  # Metrics summary

ptr/
├── SPY_ptr_20260111_114931.csv          # Point prediction for tomorrow
└── SPY_ptr_20260111_114931_summary.txt  # Tomorrow's forecast
```

### Sample Prediction Output

```
================================================================================
PREDICT TOMORROW'S RANGE - SUMMARY
================================================================================
Ticker: SPY
Prediction For: 2026-01-12

POINT PREDICTIONS
-----------------
Predicted High: $695.49 (+0.20% from close)
Predicted Low: $692.91 (-0.17% below close)
Predicted Range: $2.58

CONFIDENCE BANDS (68% CI)
-------------------------
High Upper Bound: $699.37
High Lower Bound: $694.49
Low Upper Bound: $694.49
Low Lower Bound: $688.44

TRADING LEVELS
--------------
Resistance (High Upper): $699.37
Target High: $695.49
Current: $694.07
Target Low: $692.91
Support (Low Lower): $688.44
================================================================================
```

### Best Practices

1. **Training Data:** Use 500+ days for sufficient samples
2. **Feature Health:** Monitor IV fallback % (want <50%)
3. **Model R²:** Train R² should be positive; test R² may be lower
4. **Containment:** Focus on containment metrics for trading decisions
5. **Daily Sync:** Run `python market_data_db.py --sync` before each session

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pattern_findr.git
   cd pattern_findr
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### Starting the Application
```bash
streamlit run peak_valley_ml_page.py
```

### Basic Workflow
1. Load market data (CSV or API)
2. Configure optimization parameters
3. Run optimization
4. Review results
5. Apply optimal parameters
6. Train final model
7. Evaluate performance

## Troubleshooting

### Common Issues
1. **Optimization Freezing**
   - Ensure you have sufficient system resources
   - Try reducing the number of trials or optimization time
   - Check for memory leaks in custom indicators

2. **No Trades Generated**
   - Verify input data quality
   - Adjust signal thresholds
   - Check for look-ahead bias in feature engineering

3. **Performance Issues**
   - Clear cache and restart the application
   - Reduce the size of the optimization space
   - Disable non-essential features

## Performance Considerations

### Optimization Settings
| Setting | Recommended Value | Notes |
|---------|-------------------|-------|
| Number of Trials | 100-500 | Balance between quality and runtime |
| Parallel Jobs | CPU Cores - 1 | Leave one core for system processes |
| Early Stopping | 50-100 | Stop if no improvement after N trials |
| Pruning | Enabled | Automatically stop underperforming trials |

### Memory Management
- Clear cache between optimization runs
- Use generators for large datasets
- Monitor memory usage during optimization

## Future Development

### Planned Features
1. **Enhanced Regime Detection**
   - HMM-based market regime classification
   - Adaptive strategy selection

2. **Advanced Risk Management**
   - Dynamic position sizing
   - Volatility-based stop losses

3. **Ensemble Models**
   - Combine multiple model predictions
   - Meta-learning for model selection

### Research Directions
- Reinforcement learning for trade execution
- Alternative data integration
- Explainable AI for trading signals

---

*Last Updated: January 2026*
*Version: 3.0.0*

### Changelog (v3.0.0)
- Added SQLite market data caching (`market_data_db.py`)
- Ridge regression replaces XGBoost for range prediction (73% R² improvement)
- Parallel walk-forward analysis for faster backtesting
- 10 years of historical market data support
- Parallel alpha grid search with cross-validation
