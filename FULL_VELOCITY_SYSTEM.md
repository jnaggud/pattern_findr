# Full Velocity Trading System Documentation

Complete documentation for the JD Velocity trading system - a velocity-based oscillator trading strategy with Discord alerts.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [File Reference](#file-reference)
4. [Quick Start](#quick-start)
5. [velocity_core.py - Shared Module](#velocity_corepy---shared-module)
6. [velocity_live_trader.py - Single Strategy Mode](#velocity_live_traderpy---single-strategy-mode)
7. [velocity_multi_trader.py - Multi-Strategy Mode](#velocity_multi_traderpy---multi-strategy-mode)
8. [Strategy Configuration](#strategy-configuration)
9. [State Management](#state-management)
10. [Discord Alerts](#discord-alerts)
11. [Backtest System](#backtest-system)
12. [Signal Types](#signal-types)
13. [Data Caching](#data-caching)
14. [Troubleshooting](#troubleshooting)

---

## System Overview

The Velocity Trading System uses oscillator velocity (first derivative) and acceleration (second derivative) to generate trading signals. It monitors assets like BTC-USD and SPY, sends real-time Discord alerts, and maintains trade state across restarts.

### Key Features

- **Velocity-based signals** - Trades based on oscillator momentum changes
- **Multiple signal types** - 8 different signal generation strategies
- **Discord integration** - Real-time alerts with charts to multiple servers
- **Locked backtest system** - Prevents chart repainting by locking historical trades
- **Data caching** - SQLite cache reduces API calls
- **Multi-strategy mode** - Run 6 strategies in a single process

### The 6 Core Strategies

| Strategy Name | Ticker | Lookback Period |
|---------------|--------|-----------------|
| velocity_BTC_1y | BTC-USD | 1 year |
| velocity_BTC_2y | BTC-USD | 2 years |
| velocity_BTC_5y | BTC-USD | 5 years |
| velocity_SPY_1y | SPY | 1 year |
| velocity_SPY_2y | SPY | 2 years |
| velocity_SPY_5y | SPY | 5 years |

---

## Architecture

```
                    +-------------------+
                    |  velocity_core.py |
                    |   (Shared Logic)  |
                    +-------------------+
                           /    \
                          /      \
           +-------------+        +------------------+
           | velocity_   |        | velocity_        |
           | live_trader |        | multi_trader.py  |
           | .py         |        | (Multi-Strategy) |
           | (Single)    |        +------------------+
           +-------------+

    External Dependencies:
    - oscillator_predictor_page.py (oscillator calculations)
    - novel_indicators.py (advanced oscillators)
    - data_cache.py (SQLite caching)
```

### Data Flow

1. **Data Fetch** - Price data from yfinance (with SQLite caching)
2. **Oscillator Calculation** - Composite oscillator from multiple indicators
3. **Signal Generation** - Velocity/acceleration-based entry/exit signals
4. **State Management** - JSON files track positions and trade history
5. **Alerting** - Discord webhooks for trade notifications
6. **Chart Generation** - Matplotlib charts attached to Discord messages

---

## File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `velocity_core.py` | Shared functions (data, signals, state, Discord) | ~1240 |
| `velocity_live_trader.py` | Single strategy CLI runner | ~2800 |
| `velocity_multi_trader.py` | Multi-strategy runner with interactive selection | ~575 |
| `data_cache.py` | SQLite price data caching | ~380 |

### State Files (Generated)

| File Pattern | Purpose |
|--------------|---------|
| `velocity_trade_state_{strategy}.json` | Current position, entry price/time |
| `velocity_trade_history_{strategy}.json` | Closed trade log |
| `velocity_locked_backtest_{strategy}.json` | Locked historical signals (prevents repainting) |

---

## Quick Start

### Single Strategy Mode

Run one strategy at a time with interactive selection:

```bash
python velocity_live_trader.py
```

You'll see:
```
============================================================
VELOCITY STRATEGY SELECTION
============================================================

Saved Strategies:
  1. velocity_BTC_1y_20251222_103111
      Ticker: BTC-USD | Signal: any_reversal
      Created: 2024-12-22 10:31:11

  2. velocity_SPY_2y_20251222_100334
      Ticker: SPY | Signal: any_reversal
      Created: 2024-12-22 10:03:34

Select strategy # (or press Enter for default):
```

### Multi-Strategy Mode

Run multiple strategies in one process:

```bash
python velocity_multi_trader.py
```

You'll see:
```
============================================================
VELOCITY MULTI-TRADER - Strategy Selection
============================================================

Available Strategies:
  [1] BTC-USD 1Y   - Position: None
  [2] BTC-USD 2Y   - Position: LONG @ $87234.74
  [3] BTC-USD 5Y   - Position: None
  [4] SPY 1Y       - Position: None
  [5] SPY 2Y       - Position: None
  [6] SPY 5Y       - Position: None

Enter strategy numbers (comma-separated), 'all', or 'q' to quit:
> 1,2,3
```

---

## velocity_core.py - Shared Module

The core module contains all shared logic. Both traders import from here.

### Constants

```python
# Default Discord webhook
DEFAULT_DISCORD_WEBHOOK = "https://discord.com/api/webhooks/..."

# Per-strategy webhooks for Haus Hedge server
HAUS_HEDGE_WEBHOOKS = {
    "velocity_SPY_1y": "https://discord.com/api/webhooks/...",
    "velocity_BTC_2y": "https://discord.com/api/webhooks/...",
    # ...
}

# Strategy storage directory
VELOCITY_STRATEGIES_DIR = "velocity_strategies"
```

### Key Functions

#### Data Functions

| Function | Description |
|----------|-------------|
| `fetch_price_data(ticker, days, interval)` | Fetch OHLCV data (with caching) |
| `fetch_realtime_price(ticker)` | Get current market price |

#### Oscillator Functions

| Function | Description |
|----------|-------------|
| `calculate_composite_oscillator(df, config)` | Calculate oscillator (supports novel types) |
| `calculate_velocity_signals(df, config)` | Generate buy/sell signals |

#### State Management

| Function | Description |
|----------|-------------|
| `get_state_file_path(strategy_name)` | Get path to state file |
| `load_trade_state(strategy_name)` | Load current position |
| `save_trade_state(state, strategy_name)` | Save position state |
| `log_closed_trade(...)` | Record completed trade to history |

#### Locked Backtest (Anti-Repainting)

| Function | Description |
|----------|-------------|
| `save_locked_backtest(backtest, strategy_name)` | Lock backtest results |
| `load_locked_backtest(strategy_name)` | Load locked backtest |
| `append_to_locked_backtest(entry/exit, ...)` | Add new trade to locked history |
| `detect_and_add_missed_signals(...)` | Find signals missed during downtime |

#### Discord Functions

| Function | Description |
|----------|-------------|
| `send_discord_alert(webhook, message, chart)` | Send message (posts to both servers) |
| `generate_velocity_chart(df, backtest, config)` | Generate 4-panel chart |
| `build_position_section(price, state, ...)` | Format position text |

#### Backtest Functions

| Function | Description |
|----------|-------------|
| `run_historical_backtest(df, config)` | Run full backtest |
| `run_backtest_for_period(df, config, days)` | Backtest specific period |

---

## velocity_live_trader.py - Single Strategy Mode

The original single-strategy runner. Runs independently without velocity_core.py.

### Features

- Interactive strategy selection
- Config hot-reloading
- Scheduled status updates (market open, mid-day, close)
- Hourly alerts
- Full 4-panel charts with entry/exit markers

### Usage

```bash
# Interactive selection
python velocity_live_trader.py

# Specify config directly
python velocity_live_trader.py --config path/to/config.json
```

### Main Loop

1. Fetch fresh price data every 15 minutes
2. Calculate oscillator and signals
3. Check exit conditions (if in position)
4. Check entry conditions (if no position)
5. Send scheduled status updates
6. Sleep until next cycle

### Scheduled Updates

| Event | Time (ET) | Description |
|-------|-----------|-------------|
| Market Open | 9:30 AM | Full status with chart |
| Mid-Day | 12:00 PM | Position update |
| Market Close | 4:00 PM | End-of-day summary |
| Hourly | Every hour | Brief status (if in position) |

---

## velocity_multi_trader.py - Multi-Strategy Mode

Runs multiple strategies in a single process with shared data fetches.

### Features

- Interactive strategy selection (pick specific strategies or 'all')
- Data fetched once per ticker (BTC and SPY), shared across strategies
- Independent state files per strategy
- Error isolation (one strategy failing doesn't crash others)
- Auto-disable after 10 consecutive errors

### Usage

```bash
python velocity_multi_trader.py
```

### Selection Options

```
Enter strategy numbers (comma-separated), 'all', or 'q' to quit:
> 1,2,3      # Run strategies 1, 2, and 3
> all        # Run all 6 strategies
> q          # Quit
```

### Main Loop (15-minute cycle)

```
Phase 1: Fetch Data (once per ticker)
    - BTC-USD: 200 days of daily bars
    - SPY: 200 days of daily bars
    - Real-time prices for both

Phase 2: Process Each Strategy
    - Slice data for strategy's lookback
    - Calculate oscillator and signals
    - Check exits (if in position)
    - Check entries (if no position)

Phase 3: Sleep
    - Wait until next cycle
```

### Error Handling

```python
# Per-strategy error tracking
strat['error_count'] += 1

# Auto-disable after 10 errors
if strat['error_count'] >= 10:
    strat['enabled'] = False
    print(f"DISABLED {strategy_name} after 10 errors")
```

---

## Strategy Configuration

Strategies are stored in `velocity_strategies/` as JSON bundles.

### Directory Structure

```
velocity_strategies/
  velocity_BTC_1y_20251222_103111/
    velocity_config.json
  velocity_BTC_2y_20251222_104158/
    velocity_config.json
  velocity_SPY_1y_20251222_101525/
    velocity_config.json
  ...
```

### Configuration Parameters

```json
{
  "ticker": "BTC-USD",
  "strategy_name": "velocity_BTC_1y",
  "signal_type": "any_reversal",
  "oscillator_type": "composite_smooth",

  "vel_smoothing": 3,
  "oversold_threshold": -0.3,
  "overbought_threshold": 0.3,
  "extreme_zone_mult": 1.5,

  "require_accel": true,
  "vel_threshold": 0,
  "accel_threshold": 0,

  "stop_loss_pct": 5.0,
  "take_profit_pct": 10.0,
  "exit_on_opposite_signal": true,
  "exit_on_midline_cross": false,
  "min_bars_between": 1,

  "rsi_filter": "none",
  "use_macd_confirm": false,
  "use_bb_filter": false,

  "discord_webhook": "https://discord.com/api/webhooks/..."
}
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | string | - | Asset symbol (BTC-USD, SPY, etc.) |
| `signal_type` | string | "any_reversal" | Signal generation method |
| `oscillator_type` | string | "composite_smooth" | Oscillator type |
| `vel_smoothing` | int | 3 | Smoothing window for velocity |
| `oversold_threshold` | float | -0.3 | Oversold zone threshold |
| `overbought_threshold` | float | 0.3 | Overbought zone threshold |
| `extreme_zone_mult` | float | 1.5 | Multiplier for extreme zones |
| `require_accel` | bool | true | Require acceleration confirmation |
| `stop_loss_pct` | float | 5.0 | Stop loss percentage |
| `take_profit_pct` | float | 10.0 | Take profit percentage |
| `exit_on_opposite_signal` | bool | true | Exit on opposite signal |
| `exit_on_midline_cross` | bool | false | Exit when oscillator crosses 0 |
| `min_bars_between` | int | 1 | Minimum bars between trades |

---

## State Management

### Trade State File

`velocity_trade_state_{strategy}.json`

```json
{
  "position": "long",
  "entry_price": 87234.74,
  "entry_time": "2025-12-25 00:00:00",
  "last_signal_time": "2025-12-25"
}
```

| Field | Description |
|-------|-------------|
| `position` | Current position: "long", "short", or null |
| `entry_price` | Entry price of current position |
| `entry_time` | Timestamp when position was opened |
| `last_signal_time` | Last processed signal date (prevents duplicates) |

### Trade History File

`velocity_trade_history_{strategy}.json`

```json
[
  {
    "id": 1,
    "ticker": "BTC-USD",
    "strategy_name": "velocity_BTC_2y",
    "type": "LONG",
    "entry_price": 85000.00,
    "exit_price": 92000.00,
    "entry_time": "2025-12-10 10:30:00",
    "exit_time": "2025-12-18 14:45:00",
    "exit_reason": "Take Profit",
    "pnl_pct": 8.24,
    "pnl_dollars": 7000.00
  }
]
```

### Locked Backtest File

`velocity_locked_backtest_{strategy}.json`

Prevents chart repainting by locking historical entries and exits.

```json
{
  "locked_at": "2025-01-05T10:30:00",
  "entries": [
    {"date": "2025-12-25", "price": 87234.74, "position": "long"}
  ],
  "exits": [
    {"date": "2025-12-18", "price": 92000.00, "pnl": 8.24, "reason": "Take Profit"}
  ],
  "current_position": {
    "position": "long",
    "entry_price": 87234.74,
    "entry_date": "2025-12-25"
  },
  "num_trades": 5,
  "win_rate": 60.0,
  "total_return": 25.4
}
```

---

## Discord Alerts

### Dual-Server Posting

All alerts post to two servers:
1. **Primary server** - From config webhook
2. **Haus Hedge server** - From `HAUS_HEDGE_WEBHOOKS` dictionary

### Alert Types

#### Entry Alert
```
**[BTC-USD 2Y] LONG ENTRY**
Signal: 2025-12-25
Entry: $87,234.74
```

#### Exit Alert
```
**[BTC-USD 2Y] LONG EXIT** (Take Profit)
Entry: $87,234.74 -> Exit: $95,500.00
**P&L: +9.48%** ($8,265)
```

#### Status Update
```
**[BTC-USD 2Y] Status Update**

**Position:** LONG
- Entry: $87,234.74 on 2025-12-25
- Current: $93,500.00 | **P&L: +7.18%** ($6,265)
- Duration: 10d 5h
- SL: $82,873.00 | TP: $95,958.21

**Stats:**
- Trades: 12
- Win Rate: 67%
- Return: 45.2%
- Profit Factor: 2.4
```

### Chart Panels

Generated charts include 4 panels:
1. **Price chart** - Candlesticks with entry/exit markers
2. **JD Oscillator** - Composite oscillator with zones
3. **JD Signal** - Velocity and acceleration lines
4. **Equity curve** - Cumulative P&L

---

## Backtest System

### Historical Backtest

On startup, both traders run a historical backtest to:
1. Determine if a position should be open
2. Calculate performance statistics
3. Generate the initial locked backtest

### Locked Backtest (Anti-Repainting)

The locked backtest system prevents chart repainting:

1. **Initial lock** - First run locks all historical signals
2. **Append-only** - New trades are appended, never recalculated
3. **Missed signal detection** - Signals during downtime marked as "missed"
4. **Chart markers** - Locked trades show as solid markers, missed as orange

### Missed Signal Detection

When the bot restarts after downtime:

```python
def detect_and_add_missed_signals(fresh_backtest, strategy_name):
    """
    Compares fresh backtest to locked backtest.
    Any signals after the last locked date are marked as 'missed'.
    """
```

---

## Signal Types

The system supports 8 signal generation methods:

| Signal Type | Description | Aggressiveness |
|-------------|-------------|----------------|
| `velocity_crossover_and_zone` | Velocity crosses zero AND in oversold/overbought zone | Conservative |
| `velocity_crossover_or_zone` | Velocity crosses zero OR in extreme zone | Balanced |
| `zone_only` | Only trade from extreme zones | Conservative |
| `momentum` | Strong momentum in opposite zone | Moderate |
| `any_reversal` | Velocity crossover OR extreme zone OR momentum | Aggressive |
| `double_bottom` | Two velocity crossovers in oversold zone | Conservative |
| `divergence` | Price/oscillator divergence in zone | Moderate |
| `breakout` | Oscillator breaks out of zone | Moderate |

### Signal Calculation

```python
# Velocity = first derivative of oscillator
df['velocity'] = df['osc_smooth'].diff()

# Acceleration = second derivative
df['acceleration'] = df['velocity'].diff()

# Velocity zero-crossings
df['vel_cross_up'] = (df['velocity'] > 0) & (df['velocity'].shift(1) <= 0)
df['vel_cross_down'] = (df['velocity'] < 0) & (df['velocity'].shift(1) >= 0)
```

---

## Data Caching

The system uses SQLite caching to reduce API calls.

### Cache Location

```
price_data.db  (SQLite database in project root)
```

### Cache Management

```bash
# List cached tickers
python data_cache.py list

# Fetch and cache data
python data_cache.py fetch BTC-USD 365

# Clear cache
python data_cache.py clear           # All
python data_cache.py clear BTC-USD   # Specific ticker
```

### Cache Strategy

1. **Fresh cache (< 1 day old)** - Use cached data
2. **Stale cache (1-7 days)** - Fetch only recent data, append to cache
3. **Old cache (> 7 days)** - Full refresh
4. **API failure** - Fall back to cached data

---

## Troubleshooting

### Common Issues

#### "No data returned for ticker"

```bash
# Check cache
python data_cache.py list

# Force refresh
python data_cache.py clear BTC-USD
python data_cache.py fetch BTC-USD 365
```

#### "Config not found"

Ensure strategy directory exists:
```bash
ls velocity_strategies/
```

#### Position not syncing

Check state file:
```bash
cat velocity_trade_state_velocity_BTC_2y.json
```

Reset state:
```bash
rm velocity_trade_state_velocity_BTC_2y.json
```

#### Discord alerts not sending

Test webhook:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"content": "Test message"}' \
  "https://discord.com/api/webhooks/YOUR_WEBHOOK"
```

### Log Locations

All output goes to stdout. Redirect to file:
```bash
python velocity_multi_trader.py > velocity.log 2>&1 &
```

### Running in Background

```bash
# Using nohup
nohup python velocity_multi_trader.py > velocity.log 2>&1 &

# Using screen
screen -S velocity
python velocity_multi_trader.py
# Ctrl+A, D to detach

# Using tmux
tmux new -s velocity
python velocity_multi_trader.py
# Ctrl+B, D to detach
```

---

## Best Practices

### Strategy Management

1. **Use descriptive names** - Include ticker and lookback in strategy name
2. **Don't modify locked backtests** - They prevent repainting
3. **Review missed signals** - Check for patterns in downtime signals

### Production Deployment

1. **Use screen/tmux** - Keep process alive after SSH disconnect
2. **Monitor logs** - Watch for repeated errors
3. **Set up alerts** - Discord notifications for failures

### Performance

1. **Use data caching** - Reduces API calls and startup time
2. **Run multi-trader** - More efficient than 6 separate processes
3. **Check interval** - 15 minutes is usually sufficient for daily strategies

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-05 | 2.0 | Added velocity_core.py and velocity_multi_trader.py |
| 2024-12-29 | 1.5 | Added locked backtest system |
| 2024-12-22 | 1.0 | Initial velocity_live_trader.py |
