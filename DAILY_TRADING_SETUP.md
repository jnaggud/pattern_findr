# 📈 Daily Automated Signal Generator - Setup Guide

## Overview

This system generates **BUY/SELL/HOLD signals** for your portfolio every day using your optimized strategies. You'll receive clear trading recommendations that you can execute manually.

## 🎯 What This System Does

1. ✅ Runs daily at market open and/or close
2. ✅ Scans your portfolio of tickers
3. ✅ Uses your best-performing saved strategies
4. ✅ Generates BUY/SELL/HOLD signals for each ticker
5. ✅ Saves reports (CSV + JSON)
6. ✅ Logs all activity
7. ✅ Works with free yfinance data

## 📁 Files Created

- **`daily_signal_generator.py`** - Main signal generator script
- **`portfolio_config.json`** - Your portfolio configuration
- **`run_daily_signals.sh`** - Shell script to run with logging
- **`daily_signals/`** - Output directory for signal reports
- **`logs/`** - Log files directory

---

## 🚀 Quick Start

### Step 1: Configure Your Portfolio

Edit `portfolio_config.json`:

```json
{
  "tickers": [
    "SPY",    // Add your tickers here
    "MSTY",
    "MSTR",
    "AAPL"
  ],
  "max_positions": 5,  // Max concurrent positions
  "signal_threshold": 0.7
}
```

### Step 2: Optimize and Save Strategies

1. Open Streamlit app
2. Run optimization for your tickers
3. **Save your best strategies** (click the checkboxes)
4. System will automatically use the best strategy for each ticker

### Step 3: Test Manual Run

```bash
cd /Users/jeffersonduggan/Documents/Pattern_FindR
conda activate pattern_findr
python daily_signal_generator.py
```

You should see output like:

```
🚀 DAILY SIGNAL GENERATOR - 2025-11-10 09:30:00
============================================================
📊 Loaded portfolio with 5 tickers
============================================================
🎯 Analyzing SPY
📥 Downloading SPY data...
✅ Downloaded 252 days of SPY data
📊 Calculating indicators...
✅ Loaded strategy: MACD_RSI_BB (45.3% return)

🟢 SIGNAL: BUY
   Price: $450.23 (+1.2% today)
   Strategy: MACD_RSI_BB
   Historical Return: 45.3%
============================================================
```

### Step 4: Set Up Automated Daily Runs

Make the script executable:

```bash
chmod +x run_daily_signals.sh
```

---

## ⏰ Scheduling Options

### Option 1: Run at Market Open (9:30 AM ET)

Add to crontab for **before market open** analysis:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 9:00 AM ET every weekday)
0 9 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh
```

### Option 2: Run at Market Close (4:00 PM ET)

For **end-of-day** signals:

```bash
# Add this line (runs at 4:15 PM ET every weekday)
15 16 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh
```

### Option 3: Run BOTH Times

For **twice-daily** signals:

```bash
# Morning scan (9:00 AM ET)
0 9 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh

# Afternoon scan (4:15 PM ET)
15 16 * * 1-5 /Users/jeffersonduggan/Documents/Pattern_FindR/run_daily_signals.sh
```

### Check Your Crontab

```bash
crontab -l
```

---

## 📊 Reading the Signals

### Signal Output Example

After running, check the latest signals:

```bash
# View today's signals
cat daily_signals/signals_2025-11-10.csv
```

**CSV Format:**
```csv
ticker,date,signal,price,price_change_1d,price_change_5d,strategy_name,strategy_return
SPY,2025-11-10,BUY,450.23,1.2,3.5,MACD_RSI_BB,45.3
MSTY,2025-11-10,SELL,12.45,-2.1,-5.2,PPO_STOCH,38.7
MSTR,2025-11-10,HOLD,180.50,0.5,8.3,AROON_CMO,52.1
```

### Signal Types

- **🟢 BUY** - Strategy indicates entry point
- **🔴 SELL** - Strategy indicates exit point  
- **⚪ HOLD** - No action recommended

---

## 📈 Daily Workflow

### Morning Routine (9:00 AM)

1. **Automated scan runs** (if scheduled)
2. **Check output:**
   ```bash
   cat daily_signals/signals_$(date +%Y-%m-%d).csv
   ```
3. **Review BUY signals** - Consider opening positions
4. **Review SELL signals** - Consider closing positions
5. **Execute trades manually** in your broker

### Evening Routine (4:15 PM)

1. **Automated scan runs** (if scheduled)
2. **Review day's signals**
3. **Prepare orders for tomorrow**
4. **Update portfolio notes**

---

## 🔧 Advanced Usage

### Custom Strategy Selection

To use a specific strategy instead of auto-selecting best:

```json
{
  "default_strategy": "strategy_20251110_123456.json",
  "tickers": ["SPY", "MSTY"]
}
```

### Add More Tickers

Simply edit `portfolio_config.json`:

```json
{
  "tickers": [
    "SPY", "QQQ", "IWM",
    "MSTY", "MSTR", "TSLA",
    "AAPL", "GOOGL", "NVDA"
  ]
}
```

### View Historical Signals

```bash
# List all signal files
ls -lh daily_signals/

# View specific date
cat daily_signals/signals_2025-11-01.csv
```

### Check Logs

```bash
# View latest log
tail -100 logs/signals_*.log | tail -100

# View all today's logs
cat logs/signals_$(date +%Y-%m-%d)*.log
```

---

## 📧 Optional: Email Notifications

To receive daily signal emails, update `portfolio_config.json`:

```json
{
  "notification_email": "your.email@example.com"
}
```

Then modify `daily_signal_generator.py` to add email functionality (requires email setup).

---

## 🛠️ Troubleshooting

### No Signals Generated

**Cause:** No saved strategies found

**Fix:**
1. Open Streamlit app
2. Run optimization
3. Save at least one strategy

### "Strategy Not Found" Error

**Cause:** Saved strategies were deleted or moved

**Fix:**
```bash
# Check saved strategies
ls -lh saved_strategies/

# Re-run optimization if needed
```

### Cron Job Not Running

**Cause:** Script permissions or path issues

**Fix:**
```bash
# Make script executable
chmod +x run_daily_signals.sh

# Test manually first
./run_daily_signals.sh

# Check cron logs
grep CRON /var/log/system.log
```

### Data Download Errors

**Cause:** yfinance rate limiting or network issues

**Fix:**
- Wait a few minutes and retry
- Check internet connection
- yfinance is free but has rate limits

---

## 📊 Performance Tracking

### Create a Trade Journal

Track your manual trades alongside signals:

```bash
# Copy today's signals to your journal
cp daily_signals/signals_$(date +%Y-%m-%d).csv ~/TradingJournal/
```

### Compare Signals vs. Actual Performance

Review historical signals against actual market performance:

```python
import pandas as pd
import glob

# Load all historical signals
signal_files = glob.glob('daily_signals/signals_*.csv')
all_signals = pd.concat([pd.read_csv(f) for f in signal_files])

# Filter for specific ticker
spy_signals = all_signals[all_signals['ticker'] == 'SPY']
print(spy_signals)
```

---

## 🎯 Example Daily Output

```
🚀 DAILY SIGNAL GENERATOR - 2025-11-10 09:00:00
============================================================
📊 Loaded portfolio with 5 tickers

============================================================
🎯 Analyzing SPY
📥 Downloading SPY data...
✅ Downloaded 252 days of SPY data
📊 Calculating indicators...
✅ Loaded strategy: MACD_12_26_9 + RSI_14 (45.3% return)

🟢 SIGNAL: BUY
   Price: $450.23 (+1.2% today)
   Strategy: MACD_12_26_9 + RSI_14
   Historical Return: 45.3%

============================================================
🎯 Analyzing MSTY
📥 Downloading MSTY data...
✅ Downloaded 252 days of MSTY data
📊 Calculating indicators...
✅ Loaded strategy: PPO_12_26_9 + STOCH (38.7% return)

🔴 SIGNAL: SELL
   Price: $12.45 (-2.1% today)
   Strategy: PPO_12_26_9 + STOCH
   Historical Return: 38.7%

============================================================
📋 TODAY'S TRADING SIGNALS SUMMARY
============================================================

🟢 BUY Signals: 2
   • SPY: $450.23 (+1.2% today)
     Strategy: MACD_12_26_9 + RSI_14 (45.3% hist. return)
   • AAPL: $185.50 (+0.8% today)
     Strategy: AROON_14 + CMO_14 (41.2% hist. return)

🔴 SELL Signals: 1
   • MSTY: $12.45 (-2.1% today)
     Strategy: PPO_12_26_9 + STOCH (38.7% hist. return)

⚪ HOLD Signals: 2
   • MSTR: $180.50 (+0.5% today)
   • NVDA: $495.30 (+2.1% today)

============================================================
💡 Manual Action Required:
   • Consider BUYING: SPY, AAPL
   • Consider SELLING: MSTY
============================================================

💾 Saved signals to: daily_signals/signals_2025-11-10.csv
💾 Saved signals to: daily_signals/signals_2025-11-10.json
```

---

## 🚀 Next Steps

1. ✅ **Test the system** - Run manually first
2. ✅ **Schedule cron job** - Automate daily runs
3. ✅ **Set up trade journal** - Track your executions
4. ✅ **Monitor performance** - Compare signals vs. results
5. ✅ **Refine strategies** - Re-optimize periodically

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review signal outputs in `daily_signals/`
3. Test with manual run first
4. Verify saved strategies exist

---

**Happy Trading! 📈**
