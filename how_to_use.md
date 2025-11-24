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

**Happy Trading! 🚀📈**
