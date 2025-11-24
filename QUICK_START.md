# 🚀 Daily Signal Generator - Quick Start

## What You Got

A **fully automated daily trading signal system** that:
- ✅ Uses **free yfinance data** (no paid API needed)
- ✅ Runs **automatically at market open/close**
- ✅ Generates **BUY/SELL/HOLD signals** for your portfolio
- ✅ Saves **reports and logs** automatically
- ✅ You **manually execute trades** (no risky auto-trading)

---

## 3-Step Setup (5 minutes)

### Step 1: Configure Your Portfolio
```bash
# Edit this file to add your tickers
nano portfolio_config.json
```

Change the tickers list:
```json
{
  "tickers": ["SPY", "MSTY", "MSTR", "AAPL", "NVDA"]
}
```

### Step 2: Save Some Strategies
```bash
# Open Streamlit app
/opt/anaconda3/envs/pattern_findr/bin/streamlit run app.py --server.port=8503
```

1. Run optimization
2. **Check the boxes** to save strategies
3. Close app

### Step 3: Test It!
```bash
# Run test to verify setup
python test_signal_generator.py

# Run manual scan
python daily_signal_generator.py
```

---

## Daily Usage

### Morning (Before Market)
```bash
python daily_signal_generator.py
```

**Output shows:**
- 🟢 BUY signals → Consider opening positions
- 🔴 SELL signals → Consider closing positions
- ⚪ HOLD signals → Do nothing

### Read Today's Signals
```bash
cat daily_signals/signals_$(date +%Y-%m-%d).csv
```

---

## Automation (Optional)

### Schedule Daily Runs

```bash
# Edit crontab
crontab -e

# Run at 9:00 AM ET every weekday
0 9 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh

# Run at 4:15 PM ET every weekday (market close)
15 16 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh
```

Save and exit. Done! ✅

---

## Example Output

```
🚀 DAILY SIGNAL GENERATOR - 2025-11-10 09:00:00
============================================================

🎯 Analyzing SPY
🟢 SIGNAL: BUY
   Price: $450.23 (+1.2% today)
   Strategy: MACD_RSI_BB (45.3% hist. return)

🎯 Analyzing MSTY
🔴 SIGNAL: SELL
   Price: $12.45 (-2.1% today)
   Strategy: PPO_STOCH (38.7% hist. return)

============================================================
💡 Manual Action Required:
   • Consider BUYING: SPY
   • Consider SELLING: MSTY
============================================================
```

---

## Files & Directories

| File/Directory | Purpose |
|---------------|---------|
| `daily_signal_generator.py` | Main signal generator |
| `portfolio_config.json` | Your ticker list |
| `run_daily_signals.sh` | Automation script |
| `daily_signals/` | Signal reports (CSV + JSON) |
| `logs/` | Run logs |
| `saved_strategies/` | Your optimized strategies |

---

## Quick Commands

```bash
# Test setup
python test_signal_generator.py

# Run manual scan
python daily_signal_generator.py

# View today's signals
cat daily_signals/signals_$(date +%Y-%m-%d).csv

# View latest log
ls -lt logs/ | head -5

# Edit portfolio
nano portfolio_config.json

# List saved strategies
ls -lh saved_strategies/
```

---

## Troubleshooting

**No signals generated?**
→ Make sure you have saved strategies (run optimization in Streamlit first)

**"Module not found" errors?**
→ Activate conda environment: `conda activate pattern_findr`

**Cron not running?**
→ Make script executable: `chmod +x run_daily_signals.sh`

---

## Next Steps

1. ✅ Test with `python test_signal_generator.py`
2. ✅ Run manually once to verify
3. ✅ Set up cron job for automation
4. ✅ Check signals daily and execute trades manually
5. ✅ Keep a trade journal to track performance

---

## Full Documentation

See `DAILY_TRADING_SETUP.md` for complete guide.

---

**Happy Trading! 📈**
