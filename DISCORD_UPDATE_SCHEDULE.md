# Discord Update Schedule

## Stock Tickers (SPY, QQQ, IWM, ES=F, etc.)

All times in **Eastern Time (ET)**. For Central Time (CST), subtract 1 hour.

| Time (ET) | Time (CST) | Update Name | Description |
|-----------|------------|-------------|-------------|
| 08:30-08:45 | 07:30-07:45 | Market Open Update | Pre-market summary |
| 09:00-09:14 | 08:00-08:14 | 9:00 ET Market Update | Hourly |
| 10:00-10:14 | 09:00-09:14 | 10:00 ET Market Update | Hourly |
| 11:00-11:15 | 10:00-10:15 | Mid-Day Update | Mid-session check |
| 12:00-12:14 | 11:00-11:14 | 12:00 ET Market Update | Hourly |
| 13:00-13:14 | 12:00-12:14 | 13:00 ET Market Update | Hourly |
| 14:00-14:14 | 13:00-13:14 | 14:00 ET Market Update | Hourly |
| 15:00-15:15 | 14:00-14:15 | Afternoon Update | 1 hour before close |
| 16:00-16:15 | 15:00-15:15 | Market Close Update | At market close |
| 16:30-17:00 | 15:30-16:00 | End of Day Summary | Final daily summary |

---

## Crypto Tickers (BTC-USD, ETH-USD, etc.)

Crypto uses **6-hourly updates** since markets are 24/7.

| Time (Local) | Update Name | Description |
|--------------|-------------|-------------|
| 00:00-00:45 | Midnight Update | Daily reset |
| 06:00-06:45 | Morning Update | Morning check |
| 12:00-12:45 | Midday Update | Midday check |
| 18:00-18:45 | Evening Update | Evening check |

---

## Signal Alerts

**Trade signals are sent immediately** when detected, regardless of schedule:
- **Entry signals**: BUY/LONG alerts
- **Exit signals**: SELL/Close position alerts
- **Missed signals**: Catch-up alerts on restart

---

## Update Content

Each scheduled update includes:
1. **Full Timeframe Chart** - Entire backtest period with all trades
2. **Last 126 Days Chart** - Recent ~6 months with detailed stats

**Stats shown:**
- Current position (if any)
- Entry price & unrealized P&L
- Total trades & win rate
- Total return & profit factor
- Current oscillator/velocity readings

---

## Instance Schedule

| Channel | Ticker | Timeframe | Strategy |
|---------|--------|-----------|----------|
| spy-5yr | SPY | 5 Year | any_reversal |
| spy-2yr | SPY | 2 Year | any_reversal |
| spy-1yr | SPY | 1 Year | any_reversal |
| btc-5yr | BTC-USD | 5 Year | any_reversal |
| btc-2yr | BTC-USD | 2 Year | any_reversal |
| btc-1yr | BTC-USD | 1 Year | any_reversal |

---

*Last updated: 2026-01-15*
