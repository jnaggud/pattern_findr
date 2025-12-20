# Pattern FindR - Comprehensive Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Parameter Optimization Workflow](#parameter-optimization-workflow)
4. [Recent Enhancements](#recent-enhancements)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Performance Considerations](#performance-considerations)
9. [Future Development](#future-development)

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

*Last Updated: March 2024*
*Version: 2.0.0*
