# 📊 Understanding the Two Performance Metrics

## What Changed

The Portfolio & Daily Signals page now shows **TWO separate performance metrics** for each ticker:

1. **Performance in November 2025** (Current Month)
2. **1-Year Performance** (Full Historical)

---

## Why Two Metrics?

### Problem Before:
- Only showed one return number
- Couldn't tell if strategy was working NOW vs. historically
- Confusion: "Why is my 200% strategy showing -23%?"

### Solution Now:
- **Two metrics** show both current AND historical performance
- Instantly see if strategy is working in TODAY'S market
- Compare short-term vs. long-term viability

---

## The Two Metrics Explained

### Metric 1: Performance in November 2025 (Current Month)

**What it shows:**
- Strategy performance on **THIS MONTH'S data only**
- Tells you if the strategy works in **CURRENT market conditions**

**How it's calculated:**
- Filters data to November 1 - November 12 (current date)
- Backtests strategy on just this month's price action
- Shows return if you had traded it this month

**What it means:**
- ✅ **Positive** = Strategy is working RIGHT NOW
- ❌ **Negative** = Strategy is failing in current conditions
- ⚠️ **0%** = No trades generated this month

**Example:**
```
MSTY: Performance in November 2025: -23.6%
```
↳ If you traded this strategy in November, you'd be down 23.6%

---

### Metric 2: 1-Year Performance

**What it shows:**
- Strategy performance on **FULL YEAR of data**
- Long-term track record and reliability

**How it's calculated:**
- Backtests strategy on all available data (up to 1 year)
- Shows cumulative return over entire period
- Includes all market conditions: bull, bear, sideways

**What it means:**
- ✅ **Positive** = Strategy has edge over time
- ❌ **Negative** = Strategy doesn't work on this ticker
- 📈 **High %** = Strong historical performance

**Example:**
```
MSTY: 1-Year Performance: +115.8%
```
↳ If you had followed this strategy for a year, you'd be up 115.8%

---

## How to Interpret Both Together

### Scenario 1: ✅ Both Positive (BEST)
```
MSTY
  Performance in November 2025: +12.5%
  1-Year Performance: +115.8%
```

**Interpretation:**
- Strategy works historically ✅
- Strategy works NOW ✅
- **Action: TRADE THIS!**

**Why it's good:**
- Proven long-term edge
- Currently profitable
- Low risk of recent strategy decay

---

### Scenario 2: ⚠️ Negative Month, Positive Year (CAUTION)
```
MSTY
  Performance in November 2025: -23.6%
  1-Year Performance: +115.8%
```

**Interpretation:**
- Strategy worked great historically ✅
- Strategy failing in current conditions ❌
- **Action: WAIT or RE-OPTIMIZE**

**Why this happens:**
- Market conditions changed (bull → bear, or vice versa)
- Strategy optimized for different volatility regime
- Temporary drawdown period

**What to do:**
1. **Wait** - Strategy might recover if conditions normalize
2. **Re-optimize** - Generate new strategy on current data
3. **Skip this month** - Trade other tickers showing positive month

---

### Scenario 3: ✅ Positive Month, Negative Year (INTERESTING)
```
CONY
  Performance in November 2025: +8.2%
  1-Year Performance: -12.3%
```

**Interpretation:**
- Strategy failed historically ❌
- Strategy working NOW ✅
- **Action: TRADE CAUTIOUSLY**

**Why this happens:**
- Market regime just shifted to favor this strategy
- Previous year had different conditions
- Strategy recently started working

**What to do:**
1. **Test small** - Strategy is unproven long-term
2. **Monitor closely** - Could be temporary fit
3. **Consider re-optimizing** - Might get better strategy for current conditions

---

### Scenario 4: ❌ Both Negative (AVOID)
```
SPY
  Performance in November 2025: -5.2%
  1-Year Performance: -18.4%
```

**Interpretation:**
- Strategy doesn't work historically ❌
- Strategy doesn't work now ❌
- **Action: DON'T TRADE THIS**

**Why this happens:**
- Wrong ticker type for this strategy
- Strategy optimized on different asset (e.g., MSTY strategy on SPY)
- Incompatible price patterns

**What to do:**
1. **Skip this ticker** - Strategy not suitable
2. **Optimize separately** - Create SPY-specific strategy
3. **Remove from portfolio** - Focus on working tickers

---

## Real-World Example

### Your MSTY Strategy Results:

| Ticker | November Performance | 1-Year Performance | Action |
|--------|---------------------|-------------------|--------|
| **MSTY** | -23.6% ❌ | +115.8% ✅ | ⚠️ Wait or re-optimize |
| **CONY** | -24.5% ❌ | +95.2% ✅ | ⚠️ Wait or re-optimize |
| **MSTR** | -35.1% ❌ | +8.7% ✅ | ⚠️ Weak historically, skip |
| **SPY** | 0% ⚪ | +2.1% ✅ | ⚪ No trades, different strategy needed |

**Analysis:**
1. Your strategy worked GREAT in 2024 (all positive yearly returns)
2. November 2024 has been BAD for this strategy (all negative)
3. **Likely cause:** Market regime shift (volatility change, trend reversal)

**Recommended Action:**
1. ✅ **Re-optimize** - Create new strategy on November 2024 data
2. ✅ **Wait** - Don't trade until monthly performance turns positive
3. ✅ **Diversify** - Use multiple strategies for different conditions

---

## Decision Matrix

Use this to decide whether to trade:

| November | 1-Year | Decision | Confidence |
|----------|--------|----------|------------|
| ✅ Positive | ✅ Positive | **TRADE** | 🟢 High |
| ✅ Positive | ⚪ Neutral | **TRADE SMALL** | 🟡 Medium |
| ✅ Positive | ❌ Negative | **TRADE CAUTIOUSLY** | 🟡 Medium |
| ⚪ Neutral | ✅ Positive | **CONSIDER** | 🟡 Medium |
| ⚪ Neutral | ⚪ Neutral | **SKIP** | 🟠 Low |
| ⚪ Neutral | ❌ Negative | **SKIP** | 🔴 None |
| ❌ Negative | ✅ Positive | **WAIT/RE-OPTIMIZE** | 🟡 Medium |
| ❌ Negative | ⚪ Neutral | **SKIP** | 🔴 None |
| ❌ Negative | ❌ Negative | **AVOID** | 🔴 None |

**Legend:**
- ✅ Positive = > +5%
- ⚪ Neutral = -5% to +5%
- ❌ Negative = < -5%

---

## Best Practices

### 1. Prioritize Current Month Performance
```
Strategy working NOW > Strategy worked historically
```
- Markets change
- Recent performance = more relevant
- Don't trade negative monthly returns hoping they'll revert

### 2. Use 1-Year as Confirmation
```
Good monthly + Good yearly = High confidence
Good monthly + Bad yearly = Low confidence
```
- 1-year shows consistency
- Positive history = proven edge
- Both positive = best case

### 3. Re-optimize When Both Turn Negative
```
If your strategy shows negative for both:
→ Market conditions changed
→ Time to re-optimize
```

### 4. Don't Mix Strategy Types
```
MSTY strategy → Works on: MSTY, CONY, high-vol tickers
MSTY strategy → Fails on: SPY, stable ETFs
```
- Different tickers need different strategies
- Optimize per ticker type
- Don't force square peg in round hole

---

## FAQ

### Q: Why is my November return negative but 1-year positive?

**A:** Market conditions changed. Your strategy was optimized on 2024 data (bull market, high volatility, etc.) but November 2024 has different characteristics. This is NORMAL and expected.

**Solution:** Re-optimize on recent data or wait for conditions to improve.

---

### Q: Should I trade if November is negative but 1-year is +200%?

**A:** No. Current month performance matters more. Past success doesn't guarantee current profitability. Wait for November to turn positive or re-optimize.

---

### Q: What if November is positive but 1-year is negative?

**A:** Trade cautiously with small size. The strategy might have just started working due to regime change, but lacks long-term proof. Monitor closely.

---

### Q: Both metrics are negative. What now?

**A:** Don't trade this ticker with this strategy. Either:
1. Remove this ticker from portfolio
2. Optimize a new strategy specifically for this ticker
3. Wait and check again next month

---

### Q: How often should these metrics be updated?

**A:** 
- **Daily** - Run signal generation daily to get latest signals
- **Monthly** - Current month metric updates daily with new data
- **After optimization** - Re-generate to see new strategy performance

---

### Q: Can I ignore the 1-year metric?

**A:** Not recommended. While current month is more important for immediate trading, 1-year shows if strategy has consistent edge. Trading strategies with negative 1-year is risky even if current month is positive.

---

## Summary

### Two Metrics:
1. **November Performance** = Current viability (trade or not)
2. **1-Year Performance** = Long-term reliability (proven edge)

### Golden Rules:
1. ✅ Both positive = Strong candidate, trade it
2. ⚠️ Negative month = Don't trade, even if year is positive
3. ❌ Both negative = Avoid, strategy doesn't work on this ticker
4. 🔄 When metrics turn negative = Time to re-optimize

### Key Insight:
**Strategies decay over time as market conditions change. The two metrics help you identify when a strategy is still working vs. when it needs to be replaced.**

---

**Use both metrics together to make informed trading decisions!** 📊✅
