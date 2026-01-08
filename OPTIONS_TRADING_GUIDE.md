# Options Trading Guide - Pattern_FindR System

A complete guide to using the Pattern_FindR system to identify high-probability options trades.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Complete Workflow](#the-complete-workflow)
3. [Step 1: Generate Velocity Signals](#step-1-generate-velocity-signals)
4. [Step 2: Train Range Prediction Model](#step-2-train-range-prediction-model)
5. [Step 3: Build Options Strategy](#step-3-build-options-strategy)
6. [Step 4: Execute Trades](#step-4-execute-trades)
7. [Step 5: Manage Positions](#step-5-manage-positions)
8. [Strategy Selection Guide](#strategy-selection-guide)
9. [Risk Management Rules](#risk-management-rules)
10. [Best Practices](#best-practices)
11. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## System Overview

Pattern_FindR combines three powerful components to generate options trade recommendations:

```
+------------------+     +-------------------+     +------------------+
|  VELOCITY        |     |  RANGE            |     |  OPTIONS         |
|  TRADING SYSTEM  | --> |  PREDICTION       | --> |  BUILDER         |
|  (Step 5c)       |     |  (Price Suite)    |     |                  |
|                  |     |                   |     |                  |
|  - Direction     |     |  - Expected Move  |     |  - Strategy Type |
|  - Entry/Exit    |     |  - Confidence     |     |  - Strike/DTE    |
|  - Hold Time     |     |  - High/Low Range |     |  - Risk/Reward   |
|  - TP/SL %       |     |  - Model R²       |     |  - Max Loss      |
+------------------+     +-------------------+     +------------------+
        |                         |                        |
        v                         v                        v
   strategies/              predictions/            options_trades.json
   {TICKER}_Strategy-*      {TICKER}_range_         (trade tracking)
                           predictions.json
```

## CRITICAL: Two Data Sources Required

The Options Builder needs **BOTH** data sources to generate optimal recommendations:

| Data Source | What It Provides | How to Create |
|-------------|------------------|---------------|
| **Velocity Strategy** | Direction (Long/Short), Take Profit %, Stop Loss %, Avg Hold Time | Step 5c → Optimize → Save Strategy |
| **Range Prediction** | Expected High/Low, Expected Move %, Confidence Intervals | Daily Range Tab → Train → Save **OR** Walk Forward Tab → Run → Save |

**Without both sources, you'll see warnings and get suboptimal recommendations.**

**Key Insight**: The system tells you:
- **WHEN** to trade (velocity signals)
- **HOW MUCH** the stock will move (range prediction)
- **WHAT** options strategy to use (options builder)

---

## The Complete Workflow

### Daily Trading Routine

```
MORNING (Pre-Market):
1. Run Streamlit app
2. Check for active velocity signals
3. Review range predictions for signal tickers
4. Generate options recommendations
5. Save promising trades for execution

MARKET OPEN (9:30 AM - 10:30 AM):
6. Wait for price to stabilize (first 30 min)
7. Execute saved recommendations at target prices
8. Mark trades as executed in Position Tracker

DURING DAY:
9. Monitor open positions
10. Adjust or close if stop levels hit

END OF DAY:
11. Update position P&L
12. Review new signals for tomorrow
```

---

## Step 1: Generate Velocity Signals

### Launch the App
```bash
streamlit run oscillator_predictor_page.py
```

### Configure the Oscillator (Step 1-4)

1. **Enter Ticker**: SPY, QQQ, AAPL, etc.
2. **Load Data**: Click "Load Data" - use at least 1 year of daily data
3. **Create Composite Oscillator**: Use default weights or optimize
4. **Calculate Velocity**: Default smoothing (3) works well

### Interpret Velocity Signals

The composite oscillator velocity shows momentum:

| Signal | Velocity | Meaning | Action |
|--------|----------|---------|--------|
| BULLISH | Crossing above 0 from below | Momentum turning positive | Look for LONG trades |
| STRONG BULLISH | > +0.05 and accelerating | Strong upward momentum | Aggressive LONG |
| BEARISH | Crossing below 0 from above | Momentum turning negative | Look for SHORT trades |
| STRONG BEARISH | < -0.05 and decelerating | Strong downward momentum | Aggressive SHORT |
| NEUTRAL | Near 0, no acceleration | No clear direction | Wait or use neutral strategies |

### Optimize Strategy Parameters (Step 5c)

1. Set optimization period (180-365 days recommended)
2. Run Optuna optimization (100-500 trials)
3. Review backtest metrics:
   - **Win Rate > 60%**: Good signal quality
   - **Profit Factor > 1.5**: Profitable system
   - **Sharpe > 1.0**: Risk-adjusted returns acceptable

### Save Your Strategy

After optimization, save the strategy:
- Click "Save Strategy" in Step 5c
- This creates a folder in `strategies/` with your config
- The Options Builder will load this config

---

## Step 2: Train Range Prediction Model

### Navigate to Price Prediction Suite

Scroll down to "Price Prediction Suite" section.

### Run Walk-Forward Analysis

1. **Settings**:
   - Training Window: 180 days (6 months of training data)
   - Test Window: 30 days (1 month forward test)
   - Confidence Level: 90% (gives you prediction intervals)

2. **Click "Run Walk-Forward Analysis"**

3. **Review Results**:
   - **Model R²**: Higher is better (> 0.3 is useful)
   - **Backtest Accuracy**: % of days actual range was within prediction
   - **Predicted Range**: Tomorrow's expected high/low

### Save Predictions

**Important**: After walk-forward analysis completes:
1. Scroll to Options Trading Builder
2. If no saved predictions exist, click "Save Current Predictions"
3. This saves to `predictions/{TICKER}_range_predictions.json`

### Understanding Range Predictions

```
Example Output:
  Predicted High: $598.50 (+/- $1.20)
  Predicted Low:  $592.30 (+/- $1.15)
  Expected Move:  1.03%
  Model R²:       0.42

Interpretation:
- Stock expected to trade between $592.30 and $598.50 tomorrow
- The +/- values are 90% confidence intervals
- 1.03% expected range helps size your strikes
```

---

## Step 3: Build Options Strategy

### Navigate to Options Trading Builder

Scroll to the "Options Trading Builder" section (after Price Prediction Suite).

### Tab 1: Strategy Builder

#### Load Your Data

1. **Ticker**: Enter same ticker as your velocity analysis
2. **Contracts**: Start with 1-2 contracts until profitable
3. **Load Strategy**: Select your saved velocity strategy

#### Review Loaded Data

```
+---------------------------+  +---------------------------+
|  VELOCITY STRATEGY        |  |  RANGE PREDICTION         |
|                           |  |                           |
|  Direction: LONG          |  |  Pred High: $598.50       |
|  Take Profit: 2.1%        |  |  Pred Low:  $592.30       |
|  Stop Loss: 4.5%          |  |  Expected Move: 1.03%     |
|  Avg Hold: ~12 days       |  |  Model R²: 0.42           |
+---------------------------+  +---------------------------+
```

#### DTE Selection

The system suggests DTEs based on your strategy's average hold time:

| Hold Time | Suggested DTE | Reasoning |
|-----------|---------------|-----------|
| 5-7 days | 14-21 DTE | Buffer for theta decay |
| 10-14 days | 21-30 DTE | Standard monthly cycle |
| 15-21 days | 30-45 DTE | Extra time buffer |
| 21+ days | 45-60 DTE | Reduced theta impact |

**Rule of Thumb**: DTE = Avg Hold Days × 1.5

#### Strategy Type Selection

The system recommends based on IV and direction:

| Condition | Recommendation | Why |
|-----------|----------------|-----|
| Bullish + Low IV | Bull Call Spread | Cheap premium, defined risk |
| Bullish + High IV | Bull Put Spread (Credit) | Benefit from IV crush |
| Bearish + Low IV | Bear Put Spread | Cheap premium, defined risk |
| Bearish + High IV | Bear Call Spread (Credit) | Benefit from IV crush |
| Neutral + High IV | Iron Condor / Short Straddle | Collect premium |
| Big Move Expected + Low IV | Long Straddle/Strangle | Cheap volatility |

#### Generate Recommendations

1. Verify Target/Stop prices (auto-calculated from strategy)
2. Click "Generate Recommendations"
3. Switch to "Trade Recommendations" tab

### Tab 2: Trade Recommendations

#### Evaluate Each Trade

```
TOP PICK: Bull Call Spread $595/$600
Ticker: SPY | Direction: BULLISH

Legs:
- BUY 1x $595 CALL @ $4.50 (Exp: 2026-01-21)
- SELL 1x $600 CALL @ $2.20 (Exp: 2026-01-21)

Entry Cost:    $2.30 ($230/contract)
Max Profit:    $2.70 ($270/contract)  +117%
Max Loss:      $2.30 ($230/contract)  -100%
Break-even:    $597.30
Prob Profit:   48%
R:R Ratio:     1:1.17
DTE:           14

Notes: Max profit if SPY closes above $600 at expiration
```

#### Decision Criteria

Only take trades that meet ALL criteria:

| Metric | Minimum | Ideal |
|--------|---------|-------|
| Risk/Reward | > 1:1 | > 1:1.5 |
| Prob Profit | > 40% | > 50% |
| Max Loss | < 2% account | < 1% account |
| Break-even | Within predicted range | Near current price |

#### Save Promising Trades

Click "Save Trade" for trades you want to execute.

---

## Step 4: Execute Trades

### Pre-Execution Checklist

Before executing any trade, verify:

- [ ] Market is open and liquid (after 10:00 AM)
- [ ] No major news events today
- [ ] VIX is not spiking (< 25 ideal)
- [ ] Bid-ask spread is reasonable (< 5% of premium)
- [ ] Your velocity signal is still valid

### Execution in Robinhood

1. **Open Robinhood App**
2. **Search for ticker**
3. **Tap "Trade Options"**
4. **For Vertical Spreads**:
   - Select "Spread"
   - Choose expiration date
   - Select your strikes
   - Verify it matches your recommendation
   - Enter number of contracts

5. **Set Limit Order**:
   - Use mid-point of bid-ask
   - Be patient - don't chase

6. **Review and Submit**

### After Execution

1. Return to Streamlit app
2. Go to "Position Tracker" tab
3. Find your saved trade
4. Enter actual fill price
5. Click "Execute"

---

## Step 5: Manage Positions

### Tab 3: Position Tracker

Monitor your open positions daily:

```
OPEN POSITIONS:

SPY Bull Call Spread $595/$600
Opened: Jan 7 | DTE: 12 | Entry: $2.30 | Current: $2.80
P&L: +$50 (+22%)  [████████░░░░] 45% to target
```

### Exit Rules

| Condition | Action |
|-----------|--------|
| Hit 50% of max profit | Consider closing early |
| Hit 75% of max profit | Close position |
| DTE < 7 and profitable | Close to avoid gamma risk |
| DTE < 7 and losing | Decide: close or let expire |
| Stop loss hit | Close immediately |
| Velocity signal reverses | Close position |

### Closing Positions

1. Close in Robinhood at market or limit
2. Return to Position Tracker
3. Enter close price
4. Click "Mark Closed"
5. P&L is automatically calculated

---

## Strategy Selection Guide

### Quick Reference

| Market View | IV Level | Best Strategy |
|-------------|----------|---------------|
| Strong Bull | Low | Long Call or Bull Call Spread |
| Strong Bull | High | Bull Put Spread (Credit) |
| Mild Bull | Low | Bull Call Spread |
| Mild Bull | High | Bull Put Spread (Credit) |
| Strong Bear | Low | Long Put or Bear Put Spread |
| Strong Bear | High | Bear Call Spread (Credit) |
| Mild Bear | Low | Bear Put Spread |
| Mild Bear | High | Bear Call Spread (Credit) |
| Neutral | Low | Calendar Spread |
| Neutral | High | Iron Condor, Short Strangle |
| Breakout Expected | Low | Long Straddle/Strangle |
| Breakout Expected | High | Wait or go directional |

### IV Rank Interpretation

| IV Rank | Level | Strategy Bias |
|---------|-------|---------------|
| 0-20% | Very Low | Buy premium (long options) |
| 20-40% | Low | Slight preference to buy |
| 40-60% | Average | Either direction |
| 60-80% | High | Sell premium (credit spreads) |
| 80-100% | Very High | Definitely sell premium |

---

## Risk Management Rules

### Position Sizing

```
MAXIMUM RISK PER TRADE: 1-2% of account

Example:
  Account Size: $10,000
  Max Risk Per Trade: $200 (2%)

  If Max Loss on spread = $230
  Contracts = $200 / $230 = 0.87 → Take 1 contract

  If Max Loss on spread = $85
  Contracts = $200 / $85 = 2.35 → Take 2 contracts
```

### Portfolio Rules

| Rule | Limit |
|------|-------|
| Max positions at once | 3-5 |
| Max risk in same ticker | 4% |
| Max risk in same sector | 8% |
| Max total portfolio risk | 15% |
| Cash reserve | Always keep 50%+ |

### Stop Loss Rules

| Strategy | Stop Loss Method |
|----------|------------------|
| Long Call/Put | Close at 50% loss |
| Debit Spread | Close at 50-75% loss |
| Credit Spread | Close if underlying breaches short strike |
| Straddle/Strangle | Close at 40% loss |

---

## Best Practices

### DO:

1. **Wait for confirmation** - Don't trade the first signal, wait for velocity to confirm direction
2. **Use spreads** - Defined risk protects you from big losses
3. **Size small** - 1-2% risk max per trade
4. **Track everything** - Use the Position Tracker religiously
5. **Review weekly** - Check Performance tab, learn from losses
6. **Be patient** - Wait for high-probability setups
7. **Cut losers fast** - Don't hope, close losing trades
8. **Let winners run** - But take profits at 50-75% of max
9. **Trade liquid underlyings** - SPY, QQQ, AAPL, MSFT, etc.
10. **Avoid earnings** - Close positions before earnings announcements

### DON'T:

1. **Don't over-trade** - 2-4 trades per week is plenty
2. **Don't chase** - If you missed entry, wait for next setup
3. **Don't average down** - Accept the loss, move on
4. **Don't ignore the model** - Trust your backtested system
5. **Don't trade illiquid options** - Wide bid-ask = hidden cost
6. **Don't hold through expiration** - Close by DTE 5-7
7. **Don't revenge trade** - After a loss, take a break
8. **Don't size up after wins** - Stick to your risk rules
9. **Don't ignore IV** - High IV = sell premium, Low IV = buy premium
10. **Don't trade without a plan** - Know your entry, target, and stop before trading

---

## Common Mistakes to Avoid

### Mistake 1: Ignoring Time Decay (Theta)

**Problem**: Buying options with too little time
**Solution**: Use DTE = Hold Time × 1.5 minimum

### Mistake 2: Fighting the Trend

**Problem**: Taking bearish trades in bull markets (and vice versa)
**Solution**: Check the longer-term trend, trade with momentum

### Mistake 3: Wrong Strategy for IV

**Problem**: Buying options when IV is high (overpaying)
**Solution**: Check IV Rank - above 60% = sell premium

### Mistake 4: Position Too Large

**Problem**: One bad trade wipes out multiple winners
**Solution**: Never risk more than 2% per trade

### Mistake 5: No Exit Plan

**Problem**: Hoping losing trades recover
**Solution**: Set stop loss before entering, honor it

### Mistake 6: Overconfidence After Wins

**Problem**: Increasing size after a winning streak
**Solution**: Stick to position sizing rules always

---

## Sample Trading Plan

### Weekly Routine

```
SUNDAY EVENING:
- Review past week's trades in Performance tab
- Identify what worked and what didn't
- Scan for new velocity signals on watchlist

MONDAY MORNING:
- Run walk-forward analysis on signal tickers
- Generate options recommendations
- Save 1-2 best setups

MONDAY-WEDNESDAY:
- Execute saved trades if conditions are right
- Monitor existing positions

THURSDAY:
- Review positions approaching expiration
- Close positions with DTE < 7

FRIDAY:
- Close weekly positions
- No new trades (theta decay weekend)
- Document the week's results
```

### Trade Journal Template

For each trade, record:

```
Date: ___________
Ticker: _________
Strategy: _______________
Direction: ______________

Entry:
  Velocity Signal: ____________
  Range Prediction: High $___ Low $___
  IV Rank: ____%
  DTE: ____
  Entry Price: $____

Exit:
  Exit Date: ___________
  Exit Price: $____
  P&L: $____ (___%)

Notes:
  What went right: ________________
  What went wrong: _______________
  Lesson learned: ________________
```

---

## Quick Start Checklist

For your first trade:

- [ ] Run Streamlit app
- [ ] Enter ticker (start with SPY)
- [ ] Load 1 year of data
- [ ] Create composite oscillator
- [ ] Calculate velocity
- [ ] Check for bullish/bearish signal
- [ ] Run walk-forward analysis
- [ ] Save predictions
- [ ] Go to Options Builder
- [ ] Load your strategy
- [ ] Select 21 DTE
- [ ] Choose Vertical Spread
- [ ] Generate recommendations
- [ ] Evaluate risk/reward (must be > 1:1)
- [ ] Save trade
- [ ] Execute in Robinhood (1 contract only)
- [ ] Mark as executed in Position Tracker
- [ ] Monitor daily
- [ ] Close at 50% profit or stop loss

---

## Support

If the system generates unexpected results:
1. Check that data loaded correctly (Step 1)
2. Verify oscillator is calculating (Step 4)
3. Ensure walk-forward completed (Price Prediction Suite)
4. Check saved predictions exist (predictions/ folder)

For questions about the code: Review OPTIONS_PLAN.md

---

**Remember**: The system provides signals and recommendations. You make the final decision. Start small, track everything, and refine your approach over time.

*Good luck and trade responsibly!*
