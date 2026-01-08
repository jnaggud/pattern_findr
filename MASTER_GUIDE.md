# Pattern_FindR Master Guide

## Complete Guide to Oscillator Predictor, Testing, and Live Trading

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Oscillator Predictor Page](#2-oscillator-predictor-page)
3. [Strategy Optimization](#3-strategy-optimization)
4. [Saving & Deploying Strategies](#4-saving--deploying-strategies)
5. [Oscillator Predictor Testing Page](#5-oscillator-predictor-testing-page)
6. [Live Trader Setup](#6-live-trader-setup)
7. [Running Multiple Live Traders](#7-running-multiple-live-traders)
8. [Discord Alerts](#8-discord-alerts)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick Reference](#10-quick-reference)

---

## 1. System Overview

### Architecture Diagram

```
+------------------+     +----------------------+     +------------------+
|   Streamlit App  |     |  Strategy Configs    |     |   Live Traders   |
|                  |     |                      |     |                  |
| - Oscillator     |---->| velocity_strategies/ |---->| velocity_live_   |
|   Predictor Page |     | - SPY configs        |     | trader.py        |
| - Testing Page   |     | - BTC-USD configs    |     |                  |
+------------------+     +----------------------+     +------------------+
                                   |                          |
                                   v                          v
                         +------------------+        +------------------+
                         |  State Files     |        |  Discord Alerts  |
                         | - trade_state_   |        |  - Buy signals   |
                         |   {TICKER}.json  |        |  - Sell signals  |
                         | - trade_history_ |        |  - Status updates|
                         |   {TICKER}.json  |        +------------------+
                         +------------------+
```

### Data Flow

```
Market Data (yfinance/Polygon)
         |
         v
+------------------+
| Composite        |
| Oscillator       |
| (15+ indicators) |
+------------------+
         |
         v
+------------------+
| Velocity &       |
| Acceleration     |
| Calculation      |
+------------------+
         |
         v
+------------------+
| Signal Detection |
| (8 signal types) |
+------------------+
         |
         v
+------------------+
| Filters (RSI,    |
| MACD, BB)        |
+------------------+
         |
         v
+------------------+
| Trade Execution  |
| (Entry/Exit)     |
+------------------+
```

---

## 2. Oscillator Predictor Page

### Launching the App

```bash
cd /Users/jeffersonduggan/Documents/Pattern_FindR
streamlit run app.py
```

Then select **"Oscillator Predictor"** from the page dropdown.

### Step-by-Step Workflow

#### Step 1: Load Data

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| Ticker | Stock/crypto symbol | SPY, BTC-USD, AAPL |
| Years of Data | Historical period | 1-5 years |
| Interval | Candle timeframe | 1d (daily) |

#### Step 2: Create Composite Oscillator

The system automatically calculates a composite oscillator from 15+ indicators:

| Indicator | Weight | Description |
|-----------|--------|-------------|
| RSI | 1.5 | Relative Strength Index |
| Stochastic | 1.2 | Stochastic Oscillator |
| Williams %R | 1.0 | Williams Percent Range |
| CCI | 1.0 | Commodity Channel Index |
| MFI | 1.3 | Money Flow Index |
| ROC | 0.8 | Rate of Change |
| Momentum | 0.8 | Price Momentum |
| BB Position | 1.0 | Bollinger Band Position |
| ADX Trend | 0.5 | ADX Trend Component |

#### Step 3: Detect Peaks & Valleys

Visual display of oscillator turning points showing theoretical perfect trading performance.

#### Step 4: Velocity-Based Trading (Main Section)

This is the **primary trading system**. Configure parameters in the sidebar:

### Parameter Reference

#### Core Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Signal Type | dropdown | Entry signal algorithm |
| Velocity Smoothing | 1-10 | Smoothing period for oscillator |
| Extreme Zone Multiplier | 1.0-3.0 | Multiplier for extreme zones |
| Min Bars Between Trades | 1-10 | Minimum gap between trades |

#### Signal Types Explained

| Signal Type | Logic | Best For |
|-------------|-------|----------|
| `velocity_crossover_and_zone` | Velocity crosses zero AND in oversold/overbought | Conservative, fewer trades |
| `velocity_crossover_or_zone` | Velocity crosses zero OR in extreme zone | Balanced approach |
| `zone_only` | Extreme zone + velocity turning | Mean reversion |
| `momentum` | Strong velocity in one direction | Trend following |
| `any_reversal` | Any reversal signal (most aggressive) | Maximum opportunities |
| `double_bottom` | Two velocity crossovers in zone | Pattern-based |
| `divergence` | Price/oscillator divergence | Advanced setups |
| `breakout` | Oscillator breaks threshold | Breakout trading |

#### Threshold Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Oversold Threshold | -1.0 to 0 | Buy zone boundary |
| Overbought Threshold | 0 to 1.0 | Sell zone boundary |

#### Risk Management

| Parameter | Range | Description |
|-----------|-------|-------------|
| Stop Loss % | 0-20% | Maximum loss per trade (0 = disabled) |
| Take Profit % | 0-50% | Target profit per trade (0 = disabled) |
| Exit on Opposite Signal | checkbox | Close on reverse signal |
| Exit on Midline Cross | checkbox | Close when oscillator crosses 0 |

#### Extra Filters

| Filter | Options | Description |
|--------|---------|-------------|
| RSI Filter | none, oversold_only, overbought_only, both | RSI confirmation |
| RSI Period | 5-30 | RSI calculation period |
| RSI Oversold | 10-40 | RSI buy threshold |
| RSI Overbought | 60-90 | RSI sell threshold |
| MACD Confirm | checkbox | Require MACD confirmation |
| BB Filter | checkbox | Require Bollinger Band filter |

---

## 3. Strategy Optimization

### Smart Optimization (Optuna)

1. **Configure Optimization:**
   - Trials: 1,000 - 100,000 (more = better but slower)
   - Workers: 1-8 (parallel processing)
   - Metric: total_return, sharpe_ratio, profit_factor, win_rate

2. **Click "Run Smart Optimization"**

3. **Review Results:**
   - Top 10 parameter combinations shown
   - Click "Apply" to use best parameters
   - Review backtest chart with new parameters

### Optimization Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Total Return | Overall % gain | Higher is better |
| Sharpe Ratio | Risk-adjusted return | > 1.0 good, > 2.0 excellent |
| Profit Factor | Gross profit / Gross loss | > 1.5 good, > 2.0 excellent |
| Win Rate | % profitable trades | > 50% good |
| Max Drawdown | Largest peak-to-trough | < 20% preferred |

---

## 4. Saving & Deploying Strategies

### Save Options

```
+---------------------------+-------------------------------------------+
| Button                    | Action                                    |
+---------------------------+-------------------------------------------+
| Save Strategy Config      | Quick save to velocity_strategies/        |
| Deploy to Discord Bot     | Save + copy to production_env/            |
| Save Permanently          | Create strategy bundle with timestamp     |
+---------------------------+-------------------------------------------+
```

### Strategy File Locations

```
Pattern_FindR/
├── velocity_strategies/           # Saved strategies
│   ├── SPY_any_reversal_20251220_110939/
│   │   └── velocity_config.json
│   ├── BTC-USD_velocity_crossover_and_zone_20251221_095049/
│   │   └── velocity_config.json
│   └── velocity_BTC-USD_velocity_crossover_and_zone_JD3.json
│
├── production_env/                # Active production config
│   └── velocity_config.json
│
├── velocity_trade_state_SPY.json      # SPY position state
├── velocity_trade_state_BTC-USD.json  # BTC-USD position state
├── velocity_trade_history_SPY.json    # SPY trade history
└── velocity_trade_history_BTC-USD.json # BTC-USD trade history
```

### Saving a Strategy

1. Configure parameters (manually or via optimization)
2. Enter a **Strategy Name** (e.g., `velocity_BTC-USD_velocity_crossover_and_zone_JD3`)
3. Enter your **Discord Webhook URL**
4. Click one of the save buttons

---

## 5. Oscillator Predictor Testing Page

### Purpose

Validate strategies beyond simple backtesting with advanced testing methods.

### Available Tests

| Test | Description | Use Case |
|------|-------------|----------|
| Walk-Forward Analysis | Rolling train/test windows | Detect overfitting |
| Monte Carlo Simulation | Randomized trade order | Assess luck vs skill |
| Sensitivity Analysis | Parameter perturbation | Find robust parameters |
| Cross-Asset Testing | Test on multiple tickers | Ensure generalization |
| Market Regime Analysis | Performance by volatility | Understand conditions |
| Statistical Significance | T-tests, bootstrap | Validate edge |
| Paper Trading | Live simulation | Pre-deployment testing |

### Workflow

1. **Load Strategy:** Select a saved strategy from dropdown
2. **Select Test Period:** Choose data range
3. **Run Tests:** Execute validation methods
4. **Analyze Results:** Review charts and metrics

---

## 6. Live Trader Setup

### Prerequisites

```bash
# Ensure dependencies installed
pip install -r requirements.txt

# Required packages:
# - polygon-api-client (or yfinance fallback)
# - pandas, numpy
# - requests (for Discord)
# - matplotlib (for charts)
```

### Starting a Live Trader

#### Method 1: Interactive Selection

```bash
cd /Users/jeffersonduggan/Documents/Pattern_FindR
python velocity_live_trader.py
```

This shows an interactive menu to select from saved strategies.

#### Method 2: Direct Config Path

```bash
python velocity_live_trader.py --config velocity_strategies/SPY_any_reversal_20251220_110939/velocity_config.json -s
```

The `-s` flag skips interactive selection.

### Live Trader Output

```
============================================================
VELOCITY LIVE TRADER STARTING
============================================================
Strategy: velocity_SPY_any_reversal
Ticker: SPY
Interval: 1d
Signal Type: any_reversal
Stop Loss: 5.1%
Take Profit: 8.6%
Discord Webhook: Configured
============================================================

📅 Data period: 365 days (from config)
📊 Running historical backtest...
📁 State file: velocity_trade_state_SPY.json
📁 History file: velocity_trade_history_SPY.json

============================================================
HISTORICAL BACKTEST RESULTS
============================================================
Total Completed Trades: 23
Win Rate: 78.3%
Total Return: 45.2%
Profit Factor: 3.45
Avg Trade P&L: 1.96%
============================================================
```

---

## 7. Running Multiple Live Traders

### Architecture for Multiple Tickers

```
Terminal 1 (SPY)              Terminal 2 (BTC-USD)
+--------------------+        +--------------------+
| velocity_live_     |        | velocity_live_     |
| trader.py          |        | trader.py          |
|                    |        |                    |
| Config: SPY        |        | Config: BTC-USD    |
| State: _SPY.json   |        | State: _BTC-USD    |
+--------------------+        +--------------------+
         |                             |
         v                             v
+------------------------------------------+
|           Discord Channel               |
|  - SPY signals with [SPY] tag           |
|  - BTC-USD signals with [BTC-USD] tag   |
+------------------------------------------+
```

### Commands for Running Both

**Terminal 1 - SPY Strategy:**
```bash
cd /Users/jeffersonduggan/Documents/Pattern_FindR
python velocity_live_trader.py --config velocity_strategies/SPY_any_reversal_20251220_110939/velocity_config.json -s
```

**Terminal 2 - BTC-USD Strategy:**
```bash
cd /Users/jeffersonduggan/Documents/Pattern_FindR
python velocity_live_trader.py --config velocity_strategies/velocity_BTC-USD_velocity_crossover_and_zone_JD3.json -s
```

### State File Isolation

Each ticker gets its own state files:

| Ticker | State File | History File |
|--------|------------|--------------|
| SPY | `velocity_trade_state_SPY.json` | `velocity_trade_history_SPY.json` |
| BTC-USD | `velocity_trade_state_BTC-USD.json` | `velocity_trade_history_BTC-USD.json` |

This ensures traders don't interfere with each other.

### Using Screen/tmux for Background Running

```bash
# Using screen
screen -S spy_trader
python velocity_live_trader.py --config velocity_strategies/SPY_any_reversal_20251220_110939/velocity_config.json -s
# Press Ctrl+A, then D to detach

screen -S btc_trader
python velocity_live_trader.py --config velocity_strategies/velocity_BTC-USD_velocity_crossover_and_zone_JD3.json -s
# Press Ctrl+A, then D to detach

# List screens
screen -ls

# Reattach
screen -r spy_trader
```

---

## 8. Discord Alerts

### Alert Types

| Alert | Trigger | Content |
|-------|---------|---------|
| Startup | Bot starts | Strategy info, backtest stats, chart |
| Buy Signal | Entry condition met | Price, oscillator, velocity, chart |
| Sell Signal | Exit condition met | P&L, duration, cumulative stats, chart |
| Status Update | Scheduled times | Position status, market data, chart |

### Scheduled Updates

| Time | Update Type |
|------|-------------|
| 8:30 AM | Market Open |
| 11:00 AM | Mid-Day |
| 3:00 PM | Market Close |
| Hourly | 9AM-4PM (except 11AM, 3PM) |
| Midnight | End of Day |

### Sample Discord Message

```
📈 **[VELOCITY] BUY SIGNAL - BTC-USD**
**Strategy:** velocity_BTC-USD_velocity_crossover_and_zone_JD3
**Signal Type:** velocity_crossover_and_zone
**Signal Time:** 2025-12-21 00:00:00
**Entry Price:** $97,234.50
**Oscillator:** -0.2345
**Velocity:** 0.0234
---
📊 **Strategy Stats:**
• Trades: 23 | Win Rate: 83%
• Total Return: 86.8% | PF: 8.9
---
_SL: $95,290.81 | TP: $97,234.50_
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No config found" | Missing strategy file | Save strategy from Streamlit first |
| "Module not found" | Missing dependency | `pip install -r requirements.txt` |
| Signals don't match | Version mismatch | Re-save strategy from latest Streamlit |
| No Discord alerts | Invalid webhook | Test webhook URL, check internet |
| Wrong data period | Old config | Re-save with updated Streamlit |

### Checking Logs

```bash
# View live trader output
python velocity_live_trader.py --config <path> -s 2>&1 | tee trader.log

# Check state file
cat velocity_trade_state_SPY.json

# Check trade history
cat velocity_trade_history_SPY.json
```

### Resetting State

```bash
# Clear position (start fresh)
rm velocity_trade_state_SPY.json

# Clear history (optional)
rm velocity_trade_history_SPY.json
```

---

## 10. Quick Reference

### Essential Commands

```bash
# Start Streamlit app
streamlit run app.py

# Run SPY live trader
python velocity_live_trader.py --config velocity_strategies/SPY_any_reversal_20251220_110939/velocity_config.json -s

# Run BTC-USD live trader
python velocity_live_trader.py --config velocity_strategies/velocity_BTC-USD_velocity_crossover_and_zone_JD3.json -s

# List saved strategies
ls -la velocity_strategies/

# View strategy config
cat velocity_strategies/<strategy_name>/velocity_config.json
```

### Your Active Strategies

#### SPY Strategy: `SPY_any_reversal_20251220_110939`

| Parameter | Value |
|-----------|-------|
| Signal Type | any_reversal |
| Velocity Smoothing | 4 |
| Oversold Threshold | -0.255 |
| Overbought Threshold | 0.069 |
| Stop Loss | 5.1% |
| Take Profit | 8.6% |
| RSI Filter | overbought_only |
| Exit on Midline | Yes |

#### BTC-USD Strategy: `velocity_BTC-USD_velocity_crossover_and_zone_JD3`

| Parameter | Value |
|-----------|-------|
| Signal Type | velocity_crossover_and_zone |
| Velocity Smoothing | 3 |
| Oversold Threshold | -0.2 |
| Overbought Threshold | 0.2 |
| Stop Loss | 2.0% |
| Take Profit | 0% (disabled) |
| RSI Filter | none |
| Exit on Midline | No |

### File Locations Summary

| Purpose | Location |
|---------|----------|
| Saved Strategies | `velocity_strategies/` |
| Production Config | `production_env/velocity_config.json` |
| SPY State | `velocity_trade_state_SPY.json` |
| BTC-USD State | `velocity_trade_state_BTC-USD.json` |
| Trade History | `velocity_trade_history_{TICKER}.json` |

---

## Appendix A: Signal Type Decision Tree

```
                    +----------------+
                    | Choose Signal  |
                    | Type           |
                    +----------------+
                           |
        +------------------+------------------+
        |                  |                  |
   Conservative      Balanced          Aggressive
        |                  |                  |
        v                  v                  v
+----------------+ +----------------+ +----------------+
| velocity_      | | velocity_      | | any_reversal   |
| crossover_     | | crossover_     | | (most trades)  |
| and_zone       | | or_zone        | |                |
| (fewest trades)| | (balanced)     | |                |
+----------------+ +----------------+ +----------------+
```

---

## Appendix B: Backtest Performance Interpretation

| Metric | Poor | Average | Good | Excellent |
|--------|------|---------|------|-----------|
| Total Return | < 0% | 0-20% | 20-50% | > 50% |
| Win Rate | < 40% | 40-55% | 55-70% | > 70% |
| Profit Factor | < 1.0 | 1.0-1.5 | 1.5-2.5 | > 2.5 |
| Sharpe Ratio | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| Max Drawdown | > 30% | 20-30% | 10-20% | < 10% |

---

*Last Updated: December 2025*
*Version: 3.0 - Multi-Ticker Live Trading Support*
