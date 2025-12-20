# Pattern_FindR User Guide

## Complete Documentation for All Pages and Features

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Page Overview](#page-overview)
3. [Strategy Optimization Page](#1-strategy-optimization-page)
4. [ML Trading Signals Page](#2-ml-trading-signals-page)
5. [Peak/Valley ML Signals Page](#3-peakvalley-ml-signals-page)
6. [Peak/Valley ML v2 (Clean) Page](#4-peakvalley-ml-v2-clean-page)
7. [Oscillator Predictor Page](#5-oscillator-predictor-page)
8. [Live Trading System](#6-live-trading-system)
9. [Novel Indicators](#7-novel-indicators)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Launching the App

```bash
cd /path/to/Pattern_FindR
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Page Overview

Pattern_FindR consists of 5 main pages, accessible via the dropdown at the top:

| Page | Purpose | Best For |
|------|---------|----------|
| **Strategy Optimization** | Traditional indicator-based strategy testing | Quick backtesting, parameter tuning |
| **ML Trading Signals** | ML-based signal generation (prototype) | Experimenting with ML approaches |
| **Peak/Valley ML Signals** | Predictive ML for market timing | Learning ML trading concepts |
| **Peak/Valley ML v2 (Clean)** | Production-ready ML system | Serious ML strategy development |
| **Oscillator Predictor** | Composite oscillator + velocity trading | **Recommended** - Most powerful |

---

## 1. Strategy Optimization Page

The original page for testing traditional technical analysis strategies.

### Features

- **Ticker Selection**: Enter any stock/crypto ticker (e.g., SPY, AAPL, BTC-USD)
- **Data Period**: Select historical data range (1y, 2y, 3y, 5y)
- **Strategy Testing**: Test various indicator-based strategies
- **Backtesting**: Full backtest with trade log and performance metrics
- **Optimization**: Grid search for optimal parameters

### How to Use

1. Enter a ticker symbol in the sidebar
2. Select data period and timeframe
3. Choose indicators to include
4. Click "Run Backtest" to test the strategy
5. View results in the charts and metrics panels

### Key Metrics

- **Total Return**: Overall percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profits / Gross losses

---

## 2. ML Trading Signals Page

Machine learning-based signal generation using peak/valley detection.

### Features

- Multiple ML algorithms (Random Forest, XGBoost, LightGBM, SVM)
- Optuna hyperparameter optimization
- 130+ technical indicators as features
- Real-time signal generation

### Sidebar Configuration

| Setting | Description |
|---------|-------------|
| Detection Method | Algorithm for peak/valley detection |
| Detection Window | Days for detection (3-15) |
| Minimum Change % | Threshold for significant moves |
| Models to Train | Select ML algorithms |
| Optimization Trials | Number of Optuna trials |

### Workflow

1. Configure detection parameters in sidebar
2. Enter ticker symbol
3. Click "Load Data & Detect"
4. Train selected models
5. View predictions and backtest results

---

## 3. Peak/Valley ML Signals Page

Revolutionary predictive approach - predicts peaks/valleys 1 day in advance.

### Key Concept

- **Traditional**: "A peak happened 3 days ago" (too late)
- **This Approach**: "A peak will happen tomorrow" (actionable!)

### Features

- Predictive labeling (train on day BEFORE peaks/valleys)
- SMOTE balancing for imbalanced datasets
- Multiple ML algorithms
- Complete backtesting and performance analysis

### How to Use

1. Select training period (1y-5y recommended)
2. Enter stock ticker
3. Configure peak/valley detection parameters
4. Train models with Optuna optimization
5. Generate predictions and backtest

---

## 4. Peak/Valley ML v2 (Clean) Page

Production-ready ML system with enhanced architecture.

### Features

- Clean session state management
- SMOTE balancing for imbalanced labels
- Optuna hyperparameter optimization
- Candlestick charts with trade markers
- Equity curves and performance analysis
- Feature importance visualization
- Proper model/feature bundling for production

### Sections

#### Data Loading
- Select ticker and time period
- Load historical data with validation

#### Peak/Valley Detection
- Automatic detection of market turning points
- Configurable sensitivity

#### Feature Engineering
- 130+ technical indicators generated automatically
- Feature importance analysis

#### Model Training
- Multiple algorithm support
- Cross-validation
- Optuna optimization

#### Backtesting
- Full historical backtest
- Trade-by-trade analysis
- Equity curve visualization

---

## 5. Oscillator Predictor Page

**The most powerful and recommended page** - Combines composite oscillators with velocity-based trading.

### Step-by-Step Workflow

#### Step 1: Load Data
- Enter ticker symbol (e.g., SPY, AAPL, BTC-USD)
- Select time period and interval (daily, hourly, etc.)
- Click "Load Data"

#### Step 2: Create Composite Oscillator
- Automatically generates a composite oscillator combining:
  - RSI (Relative Strength Index)
  - Stochastic Oscillator
  - Williams %R
  - CCI (Commodity Channel Index)
  - MFI (Money Flow Index)
  - And more...

#### Step 3: Detect Peaks & Valleys
- Identifies oscillator turning points
- Shows theoretical perfect trading performance
- "Buy at valleys, sell at peaks"

#### Step 4: Create ML Features (Optional)
- Generates 100+ features for ML training
- **NEW**: Includes 8 novel indicators:
  - ARWO, DCO, VCMO, ICS, MJI, PRF, EWAF, KFIF
- Derivative features (velocity, acceleration)
- Consensus and dispersion features

#### Step 5: Train/Test Split
- Configure train/test split ratio
- Enable SMOTE for class balancing
- Enable deep learning features (optional)

#### Step 5b: Scipy Peaks Backtest
- Quick backtest using scipy peak detection
- No ML required
- Good baseline comparison

#### Step 5c: Velocity-Based Trading (Live Trading Capable)

**This is the main trading section** - Supports live deployment.

##### Trading Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| Signal Type | Type of entry signal | any_reversal |
| Oscillator Type | Which oscillator to use | composite_smooth or novel |
| Velocity Smoothing | Smoothing period | 1-5 |
| Oversold Threshold | Buy zone boundary | -0.3 to -0.5 |
| Overbought Threshold | Sell zone boundary | 0.3 to 0.5 |
| Stop Loss % | Maximum loss per trade | 3-5% |
| Take Profit % | Target profit per trade | 8-15% |
| Exit on Opposite Signal | Close on reverse signal | Yes |

##### Signal Types Explained

| Signal Type | Description |
|-------------|-------------|
| `velocity_crossover_and_zone` | Velocity crosses zero AND in extreme zone |
| `velocity_crossover_or_zone` | Velocity crosses zero OR in extreme zone |
| `zone_only` | Price enters extreme zone |
| `momentum` | Strong momentum in one direction |
| `any_reversal` | Any reversal signal (recommended) |
| `double_bottom` | Two consecutive valleys |
| `divergence` | Price/oscillator divergence |
| `breakout` | Breakout from range |

##### Smart Optimization (Optuna)

1. Configure optimization settings:
   - Trials to run (1,000 - 500,000)
   - Parallel workers
   - Optimization metric (total_return, sharpe_ratio, etc.)
2. Click "Run Smart Optimization"
3. Review top parameter combinations
4. Apply best parameters or select manually

##### Novel Oscillator Types (NEW)

All 9 oscillator types are available for optimization:

| Oscillator | Description |
|------------|-------------|
| `composite_smooth` | Original weighted average (default) |
| `arwo` | Adaptive Regime-Weighted Oscillator |
| `dco` | Divergence Consensus Oscillator |
| `vcmo` | Volume-Confirmed Momentum Oscillator |
| `ics` | Indicator Convergence Score |
| `mji` | Momentum Jerk Indicator |
| `prf` | Percentile Rank Fusion |
| `ewaf` | Entropy-Weighted Adaptive Fusion |
| `kfif` | Kalman-Filtered Indicator Fusion |

##### Save & Deploy Strategy

1. After finding good parameters, click "Save Strategy"
2. Strategies are saved to `strategies/` directory
3. Use "Deploy to Production" for live trading
4. Configure Discord webhook for alerts

##### Live Trading Bot

- Start/Stop bot from the interface
- Monitors market in real-time
- Sends Discord alerts on signals
- Supports hourly check intervals

#### Step 5d: Strategy Discovery Engine

Systematic strategy discovery using:
- 100+ technical indicators via pandas_ta
- 1000+ trading conditions
- Grid search through combinations
- ML feature importance analysis

**How to Use:**
1. Configure max strategies to test
2. Set parallel workers
3. Click "Run Strategy Discovery"
4. Review discovered strategies

#### Step 6: Train ML Model (Optional)

Train ML models to predict oscillator peaks/valleys:

1. Select models to train (Random Forest, XGBoost, etc.)
2. Configure Optuna trials
3. Click "Start Optuna Optimization"
4. Review model performance metrics

#### Step 7: Model Evaluation

After training:
- View accuracy, precision, recall, F1 scores
- Confusion matrix visualization
- Feature importance chart

#### Step 8: Backtest on Test Set

Final validation:
- Backtest predictions on held-out test data
- Compare ML signals vs velocity signals
- View equity curves and trade logs

---

## 6. Live Trading System

### Velocity Live Trader

Located in `velocity_live_trader.py` - Runs as a standalone service.

#### Starting the Live Trader

```bash
python velocity_live_trader.py
```

Or from the Oscillator Predictor page:
1. Save a strategy
2. Deploy to production
3. Start the bot

#### Configuration

Production config is stored in `production_env/velocity_config.json`:

```json
{
  "strategy_type": "velocity",
  "strategy_name": "velocity_SPY_any_reversal_jd1",
  "ticker": "SPY",
  "interval": "1d",
  "polygon_api_key": "your_api_key",
  "signal_type": "any_reversal",
  "discord_webhook": "your_webhook_url"
}
```

#### Discord Alerts

The bot sends alerts including:
- Signal type (BUY/SELL)
- Entry/exit prices
- Current position status
- Performance metrics
- Charts (optional)

---

## 7. Novel Indicators

Pattern_FindR includes 8 sophisticated novel indicators beyond traditional technical analysis.

### Composite Oscillators

#### ARWO (Adaptive Regime-Weighted Oscillator)
- Dynamically weights momentum vs mean-reversion based on market regime
- High ADX (trending): Favors momentum indicators
- Low ADX (ranging): Favors mean-reversion indicators

#### DCO (Divergence Consensus Oscillator)
- Measures how many indicators diverge from price simultaneously
- Positive: Bearish divergence (price up, indicators down)
- Negative: Bullish divergence (price down, indicators up)

#### VCMO (Volume-Confirmed Momentum Oscillator)
- Only trusts momentum signals when volume confirms the move
- Uses OBV and Accumulation/Distribution for confirmation

### Indicator Formulas

#### ICS (Indicator Convergence Score)
- Measures statistical dispersion of normalized indicators
- Strong signals when indicators converge AND agree on direction

#### MJI (Momentum Jerk Indicator)
- Second derivative of momentum
- Detects when momentum is about to change direction

#### PRF (Percentile Rank Fusion)
- Converts indicators to historical percentile ranks
- Captures "how extreme is this reading historically?"

#### EWAF (Entropy-Weighted Adaptive Fusion)
- Weights indicators by information content (Shannon entropy)
- Low entropy = indicator at extremes = higher weight

#### KFIF (Kalman-Filtered Indicator Fusion)
- Uses Kalman filter to optimally combine noisy indicator readings
- Provides smoothed estimate with uncertainty bounds

---

## 8. Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
pip install -r requirements.txt
```

#### TensorFlow warnings
These are suppressed by default. If issues occur:
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

#### Slow optimization
- Reduce number of trials
- Use fewer parallel workers
- Consider using 'haiku' model for faster processing

#### No signals generated
- Check that data loaded correctly
- Verify oscillator values are within expected range (-1 to +1)
- Try adjusting thresholds (oversold/overbought)

#### Discord alerts not working
1. Verify webhook URL is correct
2. Check internet connection
3. Ensure bot is running (check status indicator)

### Performance Tips

1. **For faster optimization**: Use 1,000-10,000 trials initially
2. **For thorough search**: Use 50,000-100,000 trials
3. **For live trading**: Use daily interval for most reliable signals
4. **For scalping**: Use 15m or 1h intervals (higher noise)

---

## Quick Reference

### Recommended Workflow

1. **Start with Oscillator Predictor page**
2. Load SPY or your preferred ticker with 1-2 years of daily data
3. Run Smart Optimization with 10,000 trials
4. Review top strategies and apply best parameters
5. Backtest to verify performance
6. Save strategy and deploy to production
7. Monitor via Discord alerts

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main application entry point |
| `oscillator_predictor_page.py` | Oscillator Predictor page |
| `velocity_live_trader.py` | Live trading bot |
| `novel_indicators.py` | Novel indicator implementations |
| `optuna_worker.py` | Optimization worker |
| `strategy_discovery_engine.py` | Strategy discovery |

### Configuration Files

| File | Purpose |
|------|---------|
| `production_env/velocity_config.json` | Live trading configuration |
| `strategy_config.json` | Strategy parameters |
| `requirements.txt` | Python dependencies |

---

## Support

For issues or feature requests, please open an issue on GitHub.

---

*Last updated: December 2024*
*Version: 2.0 with Novel Indicators*
