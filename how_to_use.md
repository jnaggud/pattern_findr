# 📘 Pattern_FindR - How to Use

## 🎯 Quick Start Guide

### Step 1: Optimize Strategies
1. Enter a ticker (e.g., SPY, MSTY, MSTR)
2. Select timeframe (1d, 1h, etc.)
3. Select period (1y, 6mo, 3mo, 1mo)
4. Click **"Find Patterns"**
5. Wait for optimization to complete
6. Review top 3 strategies
7. Click **"💾 Save Selected"** to save the best ones

### Step 2: Build Your Portfolio
1. Click **"🎯 Trade"** in the sidebar
2. Add tickers to your portfolio with **"➕ Add"**
3. Remove tickers with **"➖ Remove"**
4. Click **"💾 Save Portfolio"** to persist changes

### Step 3: Select Strategies
1. For each ticker, select which strategy to apply
2. Strategies are sorted by return (highest first)
3. Each ticker defaults to its **highest return strategy**
4. Use **Quick Actions** for bulk changes:
   - **All → Highest Return** - Apply best strategy to all
   - **All → Most Recent** - Apply newest strategy to all
   - **Reset All** - Clear all selections

### Step 4: Generate Signals
1. Click **"🎯 Generate Signals for Portfolio"**
2. Review signals in tabs:
   - **BUY** - Enter these positions
   - **SELL** - Exit these positions
   - **HOLD** - Keep current positions
3. Click **"📈 View 30-Day Performance Chart"** to see recent trades
4. Export signals with **"📥 Download CSV"** or **"📥 Download JSON"**

---

## ⏰ When to Run the System

### 🎯 **Best Practice: After Market Close**

**Recommended time:** 4:30-5:00 PM ET (after market close)

**Why:**
- ✅ Complete daily candles (accurate data)
- ✅ Indicators calculated on finished price action
- ✅ Consistent with backtest optimization
- ✅ Prepare for next day's trading

**Daily Workflow:**
```
4:00 PM ET  → Market closes
4:30 PM ET  → Run Pattern_FindR
5:00 PM ET  → Review signals
Evening     → Plan tomorrow's trades
Next Day    → Execute at open or during day
```

### ❌ **Don't Run at Market Open**
- Today's candle hasn't formed yet
- You'd be using incomplete/yesterday's data
- Signals may be inaccurate

### ⚡ **Exception: Hourly Strategies**
If using hourly candles, you can run:
- Every hour during market hours (9:30 AM - 4:00 PM ET)
- At specific times (e.g., 10 AM, 2 PM, close)

---

## 📊 Understanding Performance Metrics

Each signal shows **three different returns**:

### 1️⃣ **Optimized (period on ticker)**
- **What:** Original return from strategy optimization
- **Example:** "Optimized (1y on MSTY): +182.6%"
- **Meaning:** Strategy earned 182.6% during the 1-year optimization period on MSTY

### 2️⃣ **Last 30 Days (Current)**
- **What:** Performance over the most recent 30 days
- **Example:** "Last 30 Days: -6.2%"
- **Meaning:** Strategy lost 6.2% in the last month on current data

### 3️⃣ **Full Year Backtest**
- **What:** How strategy performs over a full year test
- **Example:** "Full Year Backtest: +160.3%"
- **Meaning:** Strategy earned 160.3% when tested on full year

### ❓ **Why Are They Different?**

**Market conditions change!**
- Strategy optimized in October may not work in November
- Volatility, trends, and patterns shift over time
- **This is normal and expected**

### 📈 **What to Look For:**

✅ **Good Strategy:**
```
Optimized: +182% ✅
Last 30 Days: +15% ✅  
Full Year: +160% ✅
→ All positive = Robust strategy!
```

⚠️ **Caution Strategy:**
```
Optimized: +182% ✅
Last 30 Days: -6% ❌
Full Year: +160% ✅
→ Short-term dip, long-term strong (use cautiously)
```

❌ **Bad Strategy:**
```
Optimized: +182% ✅
Last 30 Days: -6% ❌
Full Year: -23% ❌
→ Market changed, strategy broken (re-optimize!)
```

---

## 📈 30-Day Performance Charts

### What You See:
- **Blue line** - Price movement over last 30 days
- **Green ▲** - Buy/entry signals
- **Red ▼** - Sell/exit signals
- **Title** - Performance % for the period

### How to Use:

**1. Validate Strategy Quality:**
```
Good entries? → Buys near bottoms ✅
Good exits?   → Sells near tops ✅
Too choppy?   → Many whipsaws ❌
```

**2. Check Trade Timing:**
```
Bought high, sold low = Bad timing ❌
Bought low, sold high = Good timing ✅
```

**3. Assess Trade Frequency:**
```
0-2 trades  = Too passive ⚠️
3-5 trades  = Good ✅
10+ trades  = Overtrading ❌
```

**4. Spot Patterns:**
```
All wins     = Lucky period (test more) ⚠️
Mixed        = Normal ✅
All losses   = Broken strategy ❌
```

### Example Analysis:

**Chart shows:**
- Oct 15: Buy at $8.50 (green ▲)
- Oct 20: Sell at $9.20 (red ▼) → +8.2% gain ✅
- Oct 25: Buy at $9.00 (green ▲)
- Nov 5: Sell at $10.50 (red ▼) → +16.7% gain ✅

**Result:** +15.2% in 30 days, 2 winning trades = **Strong strategy** ✅

---

## 🔧 Managing Saved Strategies

### View Strategies:
1. Sidebar → **"📋 View All Saved Strategies"**
2. See all strategies with filters and search

### Delete Strategies:
1. View saved strategies
2. Click **"🗑️ Delete"** on individual strategies
3. Or click **"🗑️ Delete ALL Strategies"** to start fresh

### Filter & Sort:
- **Search** by name or ticker
- **Sort** by return, date, trades, or indicators
- **Filter** by min/max return

---

## 💡 Best Practices

### 1. **Optimize Regularly**
- Re-optimize every 2-4 weeks
- Market conditions change → strategies decay
- Fresh optimization = better results

### 2. **Test Before Trading**
- Check all three performance metrics
- Review 30-day chart for quality
- Start small, scale up winners

### 3. **Diversify Strategies**
- Use different strategies per ticker
- Don't apply same strategy to everything
- Mix timeframes (1d, 1h) for different signals

### 4. **Monitor Performance**
- Run daily after market close
- Track which strategies work
- Replace underperformers quickly

### 5. **Understand Risk**
- Past performance ≠ future results
- Backtests are optimistic (no slippage/fees)
- Use stop losses in real trading

---

## 🎓 Advanced Tips

### **Per-Ticker Strategy Selection**

Instead of using one strategy for all tickers, customize:

```
SPY  → Strategy A (+150% on SPY)
MSTY → Strategy B (+180% on MSTY)
MSTR → Strategy C (+200% on MSTR)
```

**Why?** Each ticker has unique patterns. Strategies optimized specifically for that ticker work better.

### **Quick Actions**

**"All → Highest Return"**
- Sets every ticker to its best performing strategy
- Use when: Starting fresh or unsure

**"All → Most Recent"**
- Sets every ticker to most recently optimized strategy
- Use when: You just re-optimized everything

**"Reset All"**
- Clears selections, goes back to defaults
- Use when: Want to start over

### **Live Trading Mode**

When optimizing, enable **"🔴 LIVE TRADING SIMULATION MODE"**:
- Uses only last 30 days of data
- Verifies strategy works with TODAY's indicators
- Prevents using indicators that have NaN values on current date

---

## 🚨 Troubleshooting

### "No BUY/SELL signals"
- Strategy might be in HOLD mode
- Market conditions don't match strategy triggers
- Try different strategy or re-optimize

### "Negative returns on portfolio"
- Market conditions changed since optimization
- Re-optimize strategies on current data
- Check 30-day charts for recent performance

### "Unknown ticker" in strategies
- Old strategies saved before ticker tracking
- Run `python fix_old_strategies.py` to fix
- Or delete old strategies and re-optimize

### "Missing columns" error
- Cache issue or data problem
- Go to Saved Strategies → **"🔄 Clear Cache & Reload"**
- Delete `data_cache/` folder if persists

---

## 📋 Daily Checklist

**After Market Close (4:30-5:00 PM ET):**

- [ ] Run Streamlit app
- [ ] Click **"🎯 Trade"** → **"Portfolio & Daily Signals"**
- [ ] Click **"🎯 Generate Signals for Portfolio"**
- [ ] Review **BUY** signals:
  - Check 30-day performance chart
  - Verify all three returns are positive
  - Plan entries for tomorrow
- [ ] Review **SELL** signals:
  - Exit these positions tomorrow
- [ ] Review **HOLD** signals:
  - Keep current positions
- [ ] Export signals if needed
- [ ] Close app

**Next Day (9:30 AM ET):**
- [ ] Execute BUY orders
- [ ] Execute SELL orders
- [ ] Monitor positions

**Weekly:**
- [ ] Review which strategies performed well
- [ ] Replace underperforming strategies
- [ ] Consider re-optimizing if market changed

**Monthly:**
- [ ] Re-optimize all tickers
- [ ] Save new best strategies
- [ ] Update portfolio allocation

---

## 🎯 Success Metrics

**Good signs:**
- ✅ Most signals show positive 30-day returns
- ✅ Charts show clean entries at bottoms, exits at tops
- ✅ 3-5 trades per month per ticker
- ✅ Consistent returns across optimization and backtests

**Warning signs:**
- ⚠️ All signals show negative recent returns
- ⚠️ Charts show many whipsaws (rapid buy/sell)
- ⚠️ Too many or too few trades
- ⚠️ Large gap between optimized and backtest returns

**Action needed:**
- ❌ Re-optimize if returns consistently negative
- ❌ Delete old strategies and start fresh
- ❌ Try different timeframes or periods
- ❌ Adjust portfolio to different tickers

---

## 📞 Quick Reference

**Main Functions:**
- **Find Patterns** - Optimize new strategies
- **Trade** - Generate daily signals
- **Saved Strategies** - View/manage strategies
- **Portfolio Management** - Add/remove tickers

**Key Files:**
- `portfolio_config.json` - Your saved portfolio
- `saved_strategies/` - All optimized strategies
- `data_cache/` - Market data cache

**Important Pages:**
- Main page - Strategy optimization
- Trade page - Daily signals
- Saved strategies - Library management

---

## 🎓 Learning Resources

**Understanding Returns:**
- Read `THREE_METRICS_EXPLAINED.md` for detailed explanation
- Check `PER_TICKER_STRATEGIES.md` for strategy selection guide

**Technical Help:**
- `OLD_STRATEGIES_FIX.md` - Fix legacy strategies
- `FIXES_APPLIED.md` - Recent updates and changes

---

## ✅ Remember

1. **Run after market close** for accurate signals
2. **Check all three metrics** before trusting a strategy
3. **Use 30-day charts** to validate trade quality
4. **Re-optimize regularly** as markets change
5. **Start small** and scale up winners
6. **Track performance** and replace losers quickly
7. **Diversify** - different strategies per ticker

**Happy Trading!**

---
---

# Velocity Trading System (CLI Bots)

The Velocity Trading System is an automated trading bot that monitors oscillator velocity and acceleration signals. It runs in the terminal and sends Discord alerts.

---

## Quick Start

### Single Strategy Mode
```bash
python velocity_live_trader.py
```
- Interactive menu to select one strategy
- Runs continuously monitoring for signals
- Sends Discord alerts on entries/exits

### Multi-Strategy Mode
```bash
python velocity_multi_trader.py
```
- Interactive menu to select multiple strategies
- Runs all selected strategies in one process
- Shares data fetches (BTC and SPY fetched once each)

---

## The 6 Available Strategies

| Strategy | Ticker | Lookback Period | Description |
|----------|--------|-----------------|-------------|
| velocity_BTC_1y | BTC-USD | 1 year | Optimized on 1 year of BTC data |
| velocity_BTC_2y | BTC-USD | 2 years | Optimized on 2 years of BTC data |
| velocity_BTC_5y | BTC-USD | 5 years | Optimized on 5 years of BTC data |
| velocity_SPY_1y | SPY | 1 year | Optimized on 1 year of SPY data |
| velocity_SPY_2y | SPY | 2 years | Optimized on 2 years of SPY data |
| velocity_SPY_5y | SPY | 5 years | Optimized on 5 years of SPY data |

---

## Single Strategy Mode (velocity_live_trader.py)

### Starting
```bash
python velocity_live_trader.py
```

### Strategy Selection Menu
```
============================================================
VELOCITY STRATEGY SELECTION
============================================================

Saved Strategies:
  1. velocity_BTC_1y_20251222_103111
      Ticker: BTC-USD | Signal: velocity_crossover_and_zone
      Created: 2024-12-22 10:31:11

  2. velocity_SPY_5y_20251222_094920
      Ticker: SPY | Signal: velocity_crossover_and_zone
      Created: 2024-12-22 09:49:20
  ...

Enter strategy number (or 'p' for production config, 'q' to quit):
```

### What It Does
1. Loads selected strategy configuration
2. Runs historical backtest on startup
3. Syncs position state with backtest
4. Sends startup Discord alert with charts
5. Enters main loop:
   - Checks for exit conditions every 15 minutes
   - Checks for new entry signals
   - Sends scheduled status updates (market open, mid-day, close)

### Command Line Options
```bash
# Use specific config file
python velocity_live_trader.py --config path/to/config.json

# Skip interactive selection
python velocity_live_trader.py --skip-selection

# Save current config as a named bundle
python velocity_live_trader.py --save my_strategy_name
```

---

## Multi-Strategy Mode (velocity_multi_trader.py)

### Starting
```bash
python velocity_multi_trader.py
```

### Strategy Selection Menu
```
============================================================
VELOCITY MULTI-TRADER - Strategy Selection
============================================================

Available Strategies:
  [1] BTC-USD 1Y    - Position: SHORT @ $87138.14
  [2] BTC-USD 2Y    - Position: None
  [3] BTC-USD 5Y    - Position: None
  [4] SPY 1Y        - Position: LONG @ $684.83
  [5] SPY 2Y        - Position: LONG @ $674.48
  [6] SPY 5Y        - Position: LONG @ $674.48

Enter strategy numbers (comma-separated), 'all', or 'q' to quit:
> 1,4,5,6

Selected 4 strategies:
  - BTC-USD 1Y
  - SPY 1Y
  - SPY 2Y
  - SPY 5Y

Press Enter to start, or 'q' to quit:
```

### Selection Options
- Enter specific numbers: `1,3,5` or `1, 3, 5`
- Run all strategies: `all`
- Quit: `q`

### Benefits Over Running Multiple Single Instances
- Single process instead of 6 terminals
- Shared data fetches (reduces API calls)
- Centralized monitoring
- Easier to start/stop

---

## File Structure

### Strategy Configurations
```
velocity_strategies/
├── velocity_BTC_1y_20251222_103111/
│   ├── velocity_config.json    # Strategy parameters
│   └── data.parquet            # Bundled price data (optional)
├── velocity_BTC_2y_20251222_104158/
│   └── velocity_config.json
└── ...
```

### State Files (per strategy)
```
velocity_trade_state_velocity_BTC_1y.json    # Current position
velocity_trade_history_velocity_BTC_1y.json  # Closed trades log
velocity_locked_backtest_velocity_BTC_1y.json # Frozen chart markers
```

### Core Modules
```
velocity_live_trader.py   # Single strategy runner
velocity_multi_trader.py  # Multi-strategy runner
velocity_core.py          # Shared logic module
data_cache.py             # SQLite price data cache
```

---

## Configuration Parameters

Each strategy config (`velocity_config.json`) contains:

### Basic Settings
```json
{
    "ticker": "BTC-USD",
    "interval": "1d",
    "strategy_name": "velocity_BTC_1y",
    "optimization_period": "1y"
}
```

### Signal Settings
```json
{
    "signal_type": "velocity_crossover_and_zone",
    "oscillator_type": "composite_smooth",
    "vel_smoothing": 3,
    "oversold_threshold": -0.3,
    "overbought_threshold": 0.3,
    "require_accel": true
}
```

### Risk Management
```json
{
    "stop_loss_pct": 5.0,
    "take_profit_pct": 10.0,
    "exit_on_opposite_signal": true,
    "exit_on_midline_cross": false
}
```

### Discord
```json
{
    "discord_webhook": "https://discord.com/api/webhooks/..."
}
```

---

## Signal Types

| Signal Type | Description |
|-------------|-------------|
| `velocity_crossover_and_zone` | Velocity crosses zero AND in oversold/overbought zone |
| `velocity_crossover_or_zone` | Velocity crosses zero OR in extreme zone |
| `zone_only` | Only enters in extreme zones with positive velocity |
| `momentum` | Strong momentum detection |
| `any_reversal` | Most aggressive - any reversal signal |
| `double_bottom` | Two velocity crossovers in oversold zone |
| `divergence` | Price/oscillator divergence |
| `breakout` | Oscillator breaks threshold |

---

## Discord Alerts

The system sends Discord alerts for:

1. **Startup** - Strategy loaded with charts and stats
2. **Entry Signals** - BUY/SELL with entry price
3. **Exit Signals** - Exit reason, P&L, hold duration
4. **Scheduled Updates** - Market open, mid-day, close (with charts)
5. **Errors** - Any critical errors

### Dual Webhooks
Each strategy can post to two Discord servers:
- Primary webhook (from config)
- Secondary Haus Hedge webhook (if strategy name matches)

---

## State Management

### Trade State File
Tracks current open position:
```json
{
    "position": "long",
    "entry_price": 87138.14,
    "entry_time": "2024-12-29 10:00:00",
    "entry_signal_bar": "2024-12-29",
    "last_signal_time": "2024-12-29"
}
```

### Locked Backtest
Prevents chart marker repainting:
```json
{
    "locked_at": "2024-12-29T10:00:00",
    "entries": [...],
    "exits": [...],
    "current_position": {...},
    "num_trades": 16,
    "win_rate": 100.0,
    "total_return": 31.43
}
```

### Trade History
Log of all closed trades:
```json
[
    {
        "id": 1,
        "ticker": "BTC-USD",
        "type": "LONG",
        "entry_price": 85000.00,
        "exit_price": 87000.00,
        "pnl_pct": 2.35,
        "pnl_dollars": 2000.00,
        "exit_reason": "Opposite Signal"
    }
]
```

---

## Data Caching

The system uses SQLite caching (`data_cache.py`) to:
- Reduce yfinance API calls
- Speed up startup time
- Provide data when API is unavailable

Cache location: `price_data.db`

### Cache Commands
```bash
# List cached tickers
python data_cache.py list

# Fetch and cache ticker
python data_cache.py fetch BTC-USD 365

# Clear cache
python data_cache.py clear
python data_cache.py clear BTC-USD  # Specific ticker
```

---

## Troubleshooting

### "No data returned for ticker"
- Check internet connection
- yfinance may be rate limiting - wait a few minutes
- Try: `python data_cache.py fetch BTC-USD 200`

### "Config file not found"
- Verify the strategy exists in `velocity_strategies/`
- Check the config path in the error message

### Discord alerts not sending
- Verify webhook URL is valid
- Check for webhook rate limiting (wait 1-2 minutes)

### Position mismatch on startup
- The system auto-syncs with backtest on startup
- Check console output for "STATE MISMATCH" messages
- State files can be manually edited if needed

### Charts not appearing
- Check for errors in console output
- Verify matplotlib is installed: `pip install matplotlib`

---

## Stopping the Bot

- Press `Ctrl+C` to gracefully stop
- The bot will send a shutdown notification to Discord
- State is automatically saved before exit

---

## Best Practices

1. **Run in screen/tmux** - Keep running even if SSH disconnects
   ```bash
   screen -S velocity
   python velocity_multi_trader.py
   # Ctrl+A, D to detach
   # screen -r velocity to reattach
   ```

2. **Monitor logs** - Check console for errors and signals

3. **Don't edit state files while running** - Stop the bot first

4. **Backup state files** - Before major changes
   ```bash
   cp velocity_trade_state_*.json backups/
   ```

5. **Test with one strategy first** - Before running all 6

---

## Architecture

```
velocity_core.py              # Shared logic (signals, state, alerts)
       |
       +---> velocity_live_trader.py   # Single strategy CLI
       |
       +---> velocity_multi_trader.py  # Multi-strategy CLI
```

Both runners use the same core logic, ensuring consistent behavior.
