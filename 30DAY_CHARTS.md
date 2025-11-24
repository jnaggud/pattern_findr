# 📈 30-Day Performance Charts

## New Feature: Visual Performance Tracking!

Each ticker in your Portfolio & Daily Signals now has an **expandable 30-day performance chart** showing:

✅ Price action over last 30 days
✅ Buy trade markers (green triangles)
✅ Sell trade markers (red triangles)
✅ Strategy performance visualization
✅ Interactive tooltips with dates and prices

---

## What You'll See

### On Each Signal (BUY, SELL, HOLD):

```
MSTY
Price: $9.34 (-2.81%)
5-Day: -8.44%
Optimized (1y on MSTY): 182.6%
Last 30 Days (Current): -6.2%
Full Year Backtest: 160.3%

Strategy: Strategy 3: RSI_14 + MACD_12_26_9...
📊 4 trades in last 30 days

[📈 View 30-Day Performance Chart ▼]
    ↓ Click to expand
    
    [Interactive Chart Appears]
    - Price line (blue)
    - Buy markers (green ▲)
    - Sell markers (red ▼)
    - Return % in title
```

---

## Chart Features

### 1. **Price Line**
- Blue line showing price movement
- Last 30 days of data
- Interactive hover shows exact price and date

### 2. **Trade Markers**
- **Green ▲** = Buy signals
- **Red ▼** = Sell signals
- Positioned at exact entry/exit prices
- Hover shows: action, date, price

### 3. **Chart Title**
- Shows ticker name
- Shows performance: "Last 30 Days Performance: +15.2%"
- Green for positive, red for negative

### 4. **Interactive**
- Hover over any point for details
- Zoom in/out
- Pan left/right
- Download as PNG

---

## Example Charts

### Scenario 1: Profitable Strategy ✅

```
MSTY - Last 30 Days Performance: +15.2%

Price line: Upward trend
Trades:
- Oct 15: Buy at $8.50 (green ▲)
- Oct 20: Sell at $9.20 (red ▼) → +8.2% gain
- Oct 25: Buy at $9.00 (green ▲)
- Nov 5: Sell at $10.50 (red ▼) → +16.7% gain

Result: Strategy caught both moves! ✅
```

### Scenario 2: Losing Strategy ❌

```
MSTR - Last 30 Days Performance: -12.5%

Price line: Choppy sideways
Trades:
- Oct 12: Buy at $235 (green ▲)
- Oct 15: Sell at $220 (red ▼) → -6.4% loss
- Oct 20: Buy at $225 (green ▲)
- Oct 25: Sell at $215 (red ▼) → -4.4% loss

Result: Whipsawed by volatility ❌
```

### Scenario 3: No Trades ⚪

```
SPY - Last 30 Days Performance: +2.1%

Price line: Steady upward trend
Trades: None

Note: Buy-and-hold (no trades in last 30 days)

Result: Strategy didn't trigger, but price went up
```

---

## How to Use

### Step 1: Generate Signals
1. Go to Portfolio & Daily Signals
2. Select strategies for each ticker
3. Click "🎯 Generate Signals for Portfolio"
4. Wait for completion

### Step 2: View Charts
1. Navigate to signal tabs (BUY, SELL, or HOLD)
2. Find the ticker you want to analyze
3. Click "📈 View 30-Day Performance Chart"
4. Chart expands below

### Step 3: Analyze Performance
- **Check trade markers:** Are buys/sells well-timed?
- **Compare to price:** Did strategy catch the move?
- **Look for patterns:** Is strategy entering too early/late?
- **Assess frequency:** Too many trades (overtrading)?

---

## What to Look For

### Good Strategy Signals ✅

**1. Trades align with trends:**
```
Price trending up → Buys near bottoms, Sells near tops ✅
```

**2. Profitable entries/exits:**
```
Buy → Price goes up → Sell = Profit ✅
```

**3. Avoided bad periods:**
```
Price drops → No buy signal = Avoided loss ✅
```

**4. Few whipsaws:**
```
Buy → Immediate sell at loss = Whipsaw ❌
Few of these = Good ✅
```

---

### Bad Strategy Signals ❌

**1. Counter-trend trades:**
```
Price trending down → Strategy keeps buying = Losses ❌
```

**2. Late entries:**
```
Price already moved up → Then buy signal = Missed move ❌
```

**3. Early exits:**
```
Sold near bottom → Price rallied = Left gains on table ❌
```

**4. Many whipsaws:**
```
Buy → Sell at loss → Buy → Sell at loss = Death by 1000 cuts ❌
```

---

## Real Example: MSTY

### Your Screenshot Shows:
- **Ticker:** MSTY
- **Last 30 Days:** -6.2%
- **Trades:** 4 trades in last 30 days

**What the chart would show:**
```
MSTY - Last 30 Days Performance: -6.2%

Likely scenario (negative return with 4 trades):
- Multiple entries and exits
- Some winning trades, more losing trades
- OR: One big loss wiped out small gains
- Net result: -6.2%

Analysis needed:
- Were entries poorly timed?
- Did strategy hold losers too long?
- Were stops too tight (whipsawed)?
```

**Action:**
1. Open chart
2. See exactly where buys/sells happened
3. Compare to price movement
4. Decide: Is strategy broken, or just unlucky period?

---

## Use Cases

### Use Case 1: Strategy Validation
**Question:** "Is my strategy actually working?"

**Steps:**
1. Generate signals
2. Open 30-day chart
3. Check if trades align with profitable moves
4. **If yes:** Strategy working! ✅
5. **If no:** Time to re-optimize ❌

---

### Use Case 2: Comparison
**Question:** "Which strategy works best on MSTY?"

**Steps:**
1. Select Strategy A for MSTY
2. Generate signals → Check chart
3. Note: Performance and trade quality
4. Go back, select Strategy B for MSTY
5. Generate signals → Check chart
6. Compare: Which had better entries/exits?
7. **Use the winner!** ✅

---

### Use Case 3: Debugging
**Question:** "Why is my 194% strategy showing -6% on MSTY?"

**Steps:**
1. Open 30-day chart for MSTY
2. Look at trade markers
3. See: Strategy bought at tops, sold at bottoms ❌
4. **Diagnosis:** Market conditions changed
5. **Fix:** Re-optimize on current data

---

### Use Case 4: Confidence Building
**Question:** "Should I trust this BUY signal?"

**Steps:**
1. See BUY signal for MSTY
2. Open 30-day chart
3. Check recent trades:
   - **Recent trades profitable?** → Trust it ✅
   - **Recent trades losing?** → Be cautious ⚠️
4. Make informed decision

---

## Chart Interpretation Guide

### Pattern 1: Good Entries (Green ▲ at bottoms)
```
Price: ╲ ╱ ╲ ╱
Buys:  ▲   ▲    ← Buying dips ✅
```
**Meaning:** Strategy catching reversals well!

---

### Pattern 2: Good Exits (Red ▼ at tops)
```
Price: ╱ ╲ ╱ ╲
Sells:   ▼   ▼  ← Selling peaks ✅
```
**Meaning:** Strategy taking profits at right time!

---

### Pattern 3: Whipsaws (Buy→Sell→Buy quickly)
```
Price: ═══════ (sideways)
Trades: ▲▼▲▼▲▼ ← Churning ❌
```
**Meaning:** Strategy confused by choppy market!

---

### Pattern 4: Trend Following (Buy and hold)
```
Price: ╱╱╱╱╱╱╱
Trades: ▲      ▼ ← Hold trend ✅
```
**Meaning:** Strategy letting winners run!

---

## Tips

### Tip 1: Compare to Buy-and-Hold
- If no trades shown, chart still displays price
- Compare strategy return to price change
- **Strategy < Price change** = Underperforming ❌

### Tip 2: Check Trade Density
- **Too many trades:** Overtrading, high costs
- **Too few trades:** Missing opportunities
- **Sweet spot:** 3-5 trades per 30 days

### Tip 3: Look for Consistency
- Do recent trades follow same pattern?
- **Consistent wins:** Strategy robust ✅
- **Erratic:** Strategy unstable ⚠️

### Tip 4: Use with Other Metrics
```
Chart shows: Good entry/exit points ✅
30-Day return: Negative ❌

Conclusion: Good timing, but market went wrong way
Action: Wait for market to turn
```

---

## FAQ

### Q: Why is chart collapsed by default?

**A:** To keep the UI clean. With multiple tickers, showing all charts would be overwhelming. Click to expand when you want to analyze.

---

### Q: Can I download the chart?

**A:** Yes! Hover over chart → Camera icon (top right) → Download as PNG

---

### Q: What if there are no trades?

**A:** Chart still shows price line. This tells you:
- Strategy didn't trigger
- Whether you missed a move (buy-and-hold would have worked)

---

### Q: Why don't trade dates match exactly?

**A:** The chart shows the last 30 calendar days. If you optimized on a different 30-day window, trades might be different.

---

### Q: Can I see more than 30 days?

**A:** Currently just 30 days to match the "Last 30 Days Performance" metric. This keeps it consistent and focused on recent activity.

---

## Summary

| Feature | Benefit |
|---------|---------|
| **30-day price chart** | See market context |
| **Buy markers (green ▲)** | Visualize entry points |
| **Sell markers (red ▼)** | Visualize exit points |
| **Interactive tooltips** | Get exact prices/dates |
| **Return % in title** | Quick performance check |
| **Per-ticker chart** | Analyze each separately |
| **Expandable** | Clean UI, expand when needed |

---

## Result

**You can now:**
- ✅ Visualize strategy performance
- ✅ See exactly where trades occurred
- ✅ Compare entry/exit timing to price action
- ✅ Identify strategy weaknesses
- ✅ Build confidence in signals
- ✅ Make informed trading decisions

**Much better than just looking at a % return number!** 📈

---

## Quick Start

1. ✅ Refresh Streamlit
2. ✅ Go to Portfolio & Daily Signals
3. ✅ Generate signals
4. ✅ Click "📈 View 30-Day Performance Chart" on any ticker
5. ✅ Analyze your strategy's trades!

**The charts will help you understand WHY your strategy is performing the way it is!** 🎯
