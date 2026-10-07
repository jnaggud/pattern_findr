# Velocity Trading System v2

A modular, SQLite-backed trading system that runs **in parallel** with the legacy `velocity_live_trader.py`.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Migration from JSON](#migration-from-json)
4. [Running Traders](#running-traders)
5. [Configuration](#configuration)
6. [Database Schema](#database-schema)
7. [API Reference](#api-reference)
8. [Parallel Operation](#parallel-operation)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Why This System?

The new `velocity_trading` package addresses issues with the monolithic `velocity_live_trader.py`:

| Old System | New System |
|------------|------------|
| 6,700+ line monolith | Modular package (~3,000 lines across 15 files) |
| JSON file state (race conditions) | SQLite with atomic transactions |
| Single script for all strategies | Separate `DailyTrader` and `IntradayTrader` |
| Discord/state desync bugs | `PositionManager` ensures consistency |

### Architecture

```
velocity_trading/
├── core/
│   ├── database.py          # SQLite schema & connections
│   └── position_manager.py  # Atomic state management (THE KEY COMPONENT)
├── data/
│   ├── fetcher.py           # Price data (yfinance, Polygon, Databento)
│   └── market_hours.py      # Market schedules (NYSE, CME, crypto)
├── indicators/
│   ├── oscillators.py       # Composite oscillator
│   └── velocity.py          # Signal generation
├── notifications/
│   └── discord.py           # Discord webhooks
├── traders/
│   ├── base_trader.py       # Common trading loop
│   ├── daily_trader.py      # SPY, BTC daily strategies
│   └── intraday_trader.py   # ES=F, GC=F 15-minute strategies
├── migration/
│   └── json_to_sqlite.py    # Migrate from old JSON files
└── db/                      # SQLite databases (one per strategy)
```

---

## Quick Start

### 1. Test the Package

```bash
cd pattern_findr

# Verify imports work
python -c "from velocity_trading import PositionManager, DailyTrader; print('OK')"
```

### 2. Run Migration (Dry Run First)

```bash
# See what would be migrated
python -m velocity_trading.migration.json_to_sqlite --dry-run

# Actually migrate (archives JSON files, doesn't delete them)
python -m velocity_trading.migration.json_to_sqlite
```

### 3. Start a Trader

```bash
# Daily SPY trader
python -m velocity_trading.traders.daily_trader --ticker SPY --webhook YOUR_WEBHOOK_URL

# Intraday ES=F trader
python -m velocity_trading.traders.intraday_trader --ticker ES=F --interval 15m
```

---

## Migration from JSON

The migration tool converts your existing JSON files to SQLite databases.

### What Gets Migrated

| JSON File | SQLite Table |
|-----------|--------------|
| `velocity_trade_state_*.json` | `positions` (current open position) |
| `velocity_locked_backtest_*.json` | `trades` (historical trades) |
| `velocity_trade_history_*.json` | `trades` (additional records) |
| `daily_alerts_*.json` | `daily_alerts` |

### Migration Commands

```bash
# Preview migration (no changes)
python -m velocity_trading.migration.json_to_sqlite --dry-run

# Migrate all strategies
python -m velocity_trading.migration.json_to_sqlite

# Migrate single strategy
python -m velocity_trading.migration.json_to_sqlite --strategy velocity_SPY_5y

# Migrate without archiving JSON files
python -m velocity_trading.migration.json_to_sqlite --no-archive

# Verify migration
python -m velocity_trading.migration.json_to_sqlite --verify velocity_SPY_5y
```

### After Migration

- Original JSON files are moved to `velocity_trading/json_archive/`
- SQLite databases are created in `velocity_trading/db/`
- Each strategy gets its own database: `velocity_SPY_5y.db`, `velocity_ES=F_15m.db`, etc.

### Rollback

If something goes wrong:
1. Stop the new traders
2. Copy JSON files back from `velocity_trading/json_archive/`
3. Old `velocity_live_trader.py` still works unchanged

---

## Running Traders

### Daily Trader

For strategies that trade on daily bars (SPY, QQQ, BTC-USD daily).

```bash
# Basic usage
python -m velocity_trading.traders.daily_trader --ticker SPY

# With all options
python -m velocity_trading.traders.daily_trader \
    --ticker SPY \
    --strategy velocity_SPY_5y \
    --webhook https://discord.com/api/webhooks/... \
    --stop-loss 2.0 \
    --take-profit 5.0
```

**When it checks for signals:**
- Stocks (SPY, QQQ): After 4:05 PM ET until 6:00 PM ET
- Crypto (BTC-USD): After 00:00 UTC

### Intraday Trader

For strategies that trade on intraday bars (ES=F 15m, GC=F 30m).

```bash
# Basic usage
python -m velocity_trading.traders.intraday_trader --ticker ES=F --interval 15m

# With all options
python -m velocity_trading.traders.intraday_trader \
    --ticker ES=F \
    --interval 15m \
    --strategy velocity_ESF_15m \
    --webhook https://discord.com/api/webhooks/... \
    --stop-loss 1.5 \
    --take-profit 3.0
```

**When it checks for signals:**
- After each bar closes (e.g., every 15 minutes)
- Only when market is open (CME: Sun 5 PM - Fri 5 PM CT, closed Sat)

### Environment Variables

Instead of command-line arguments, you can use environment variables:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Now webhook is automatic
python -m velocity_trading.traders.daily_trader --ticker SPY
```

### Running as Background Service

```bash
# Using nohup
nohup python -m velocity_trading.traders.daily_trader --ticker SPY > spy_trader.log 2>&1 &

# Using screen
screen -S spy_trader
python -m velocity_trading.traders.daily_trader --ticker SPY
# Ctrl+A, D to detach

# Using tmux
tmux new -s spy_trader
python -m velocity_trading.traders.daily_trader --ticker SPY
# Ctrl+B, D to detach
```

---

## Configuration

### Strategy Configuration

Pass a config dict when creating traders programmatically:

```python
from velocity_trading import DailyTrader

config = {
    # Exit parameters
    'stop_loss_pct': 2.0,
    'take_profit_pct': 5.0,

    # Signal parameters
    'signal_type': 'any_reversal',  # or 'zone_exit', 'velocity_cross', 'midline_cross'
    'oversold_threshold': -0.3,
    'overbought_threshold': 0.3,

    # Timing (daily)
    'signal_window_minutes': 120,   # Check for 2 hours after close
    'check_interval_seconds': 120,  # Check every 2 minutes
}

trader = DailyTrader(
    strategy_name='velocity_SPY_5y',
    ticker='SPY',
    webhook_url='https://discord.com/api/webhooks/...',
    config=config
)

trader.run()
```

### Signal Types

| Signal Type | Description |
|-------------|-------------|
| `any_reversal` | Buy when in oversold zone AND velocity turns positive |
| `zone_exit` | Buy when exiting oversold zone |
| `velocity_cross` | Buy when velocity crosses above zero |
| `midline_cross` | Buy when oscillator crosses above zero |
| `zone_velocity` | Buy when in oversold zone AND velocity is positive |

---

## Database Schema

Each strategy has its own SQLite database with these tables:

### `trades` - Complete Trade History

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    ticker TEXT NOT NULL,

    -- Entry
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_signal_bar TEXT,
    position_type TEXT DEFAULT 'long',

    -- Exit (NULL if still open)
    exit_date TEXT,
    exit_price REAL,
    exit_signal_bar TEXT,
    exit_reason TEXT,
    pnl_pct REAL,
    pnl_dollars REAL,

    -- Metadata
    is_missed BOOLEAN DEFAULT FALSE,
    created_at TEXT
);
```

### `positions` - Current Open Position

```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT UNIQUE NOT NULL,  -- Only ONE position per strategy
    ticker TEXT NOT NULL,
    position_type TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    entry_signal_bar TEXT,
    last_signal_time TEXT,
    trade_id INTEGER REFERENCES trades(id)
);
```

### `strategy_stats` - Cached Performance

```sql
CREATE TABLE strategy_stats (
    strategy_name TEXT UNIQUE NOT NULL,
    num_trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    total_return REAL DEFAULT 0.0,
    profit_factor REAL DEFAULT 0.0,
    last_updated TEXT
);
```

### Querying the Database

```bash
# Open database
sqlite3 velocity_trading/db/velocity_SPY_5y.db

# View recent trades
SELECT entry_date, entry_price, exit_date, exit_price, pnl_pct
FROM trades ORDER BY id DESC LIMIT 10;

# Check current position
SELECT * FROM positions;

# View stats
SELECT * FROM strategy_stats;
```

---

## API Reference

### PositionManager

The core component that ensures atomic state transitions.

```python
from velocity_trading import PositionManager

pm = PositionManager('velocity_SPY_5y')

# Enter position (returns tuple: success, result)
success, result = pm.enter_position(
    ticker='SPY',
    position_type='long',
    entry_price=590.50,
    entry_date='2026-01-21T10:00:00',
    entry_signal_bar='2026-01-21 00:00:00'
)

if success:
    print(f"Entered at {result['entry_price']}")
    send_discord_alert(...)  # Only send if DB updated!
else:
    print(f"Entry rejected: {result['error']}")

# Exit position
success, result = pm.exit_position(
    exit_price=600.25,
    exit_date='2026-01-25T16:00:00',
    exit_reason='Take Profit (1.65%)'
)

if success:
    print(f"P&L: {result['pnl_pct']:.2f}%")

# Check current position
position = pm.get_current_position()
if position:
    print(f"In {position.position_type} @ {position.entry_price}")
else:
    print("Flat")

# Get trade history (for charts)
history = pm.get_entries_and_exits()
print(f"Entries: {len(history['entries'])}")
print(f"Exits: {len(history['exits'])}")

# Get stats
stats = pm.get_stats()
print(f"Win rate: {stats['win_rate']:.0f}%")
```

### TradingDatabase

Low-level database access.

```python
from velocity_trading import TradingDatabase

db = TradingDatabase('velocity_trading/db/velocity_SPY_5y.db')

# Query trades
trades = db.execute(
    "SELECT * FROM trades WHERE pnl_pct > ?",
    (2.0,)
).fetchall()

# Use transactions
with db.transaction():
    db.execute("UPDATE strategy_stats SET win_rate = ?", (85.0,))
    db.execute("UPDATE strategy_stats SET num_trades = ?", (50,))
```

---

## Parallel Operation

The new system is designed to run alongside the old system during transition.

### Running Both Systems

```bash
# Terminal 1: Old system (unchanged)
python velocity_live_trader.py --config your_config.json

# Terminal 2: New system
python -m velocity_trading.traders.daily_trader --ticker SPY
```

### Comparison Testing

Run both systems and compare:
1. Do they generate the same signals?
2. Do they send the same Discord messages?
3. Are the stats identical?

### Cutover Checklist

Before switching production to the new system:

- [ ] Migrated all JSON data to SQLite
- [ ] Ran new system in parallel for 1+ week
- [ ] Verified signals match between old and new
- [ ] Verified Discord messages are correct
- [ ] Tested restart/recovery behavior
- [ ] Documented any differences

### Rollback

If issues occur after switching:

1. Stop new traders: `pkill -f daily_trader.py`
2. Restore JSON from archive: `cp velocity_trading/json_archive/*.json .`
3. Start old trader: `python velocity_live_trader.py`

---

## Troubleshooting

### Common Issues

**Import Error: module not found**
```bash
# Make sure you're in the right directory
cd pattern_findr
python -c "from velocity_trading import PositionManager"
```

**Database locked**
```
sqlite3.OperationalError: database is locked
```
Only one process should write to a strategy's database at a time. Check if another trader is running.

**No signals detected**
- Check if market is open: `python -c "from velocity_trading.data import is_market_open; print(is_market_open('SPY'))"`
- Check signal window timing in logs
- Verify data is being fetched (look for "Checking SPY..." in output)

**Discord messages not sending**
- Verify webhook URL is correct
- Check for rate limiting (Discord limits webhooks)
- Look for "Discord error:" in logs

### Debugging

```python
# Check position state
from velocity_trading import PositionManager
pm = PositionManager('velocity_SPY_5y')
print(pm.get_current_position())
print(pm.get_stats())

# Check database directly
import sqlite3
conn = sqlite3.connect('velocity_trading/db/velocity_SPY_5y.db')
print(conn.execute("SELECT * FROM positions").fetchall())
print(conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 5").fetchall())
```

### Logs

Traders print to stdout. Redirect to file for logging:

```bash
python -m velocity_trading.traders.daily_trader --ticker SPY 2>&1 | tee -a spy_trader.log
```

---

## Next Steps

After setup:

1. **Run migration** - Convert JSON to SQLite
2. **Start in parallel** - Run new trader alongside old
3. **Monitor for 1 week** - Compare signals and behavior
4. **Switch over** - Stop old trader, rely on new system
5. **Clean up** - Remove archived JSON files when confident

---

## Support

- Check logs for error messages
- Query SQLite database directly for state
- Old system (`velocity_live_trader.py`) is always available as fallback
