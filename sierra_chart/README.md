# Sierra Chart Integration

This directory contains the ACSIL custom study for displaying Pattern_FindR signals in Sierra Chart.

## Files

- `PFR_SignalDisplay.cpp` - ACSIL custom study source code

## Installation

### 1. Copy Study to Sierra Chart

Copy `PFR_SignalDisplay.cpp` to your Sierra Chart ACS_Source folder:
- Windows: `C:\SierraChart\ACS_Source\`

### 2. Build the Study

In Sierra Chart:
1. Go to **Analysis > Build Custom Studies DLL**
2. Wait for compilation to complete
3. The study will appear as "Pattern FindR Signals"

### 3. Add to Chart

1. Open your chart (ES, GC, NQ, etc.)
2. Go to **Analysis > Studies > Add Custom Study**
3. Search for "Pattern FindR"
4. Add "Pattern FindR Signals" to your chart

### 4. Configure Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Signal Server URL | URL of the Pattern_FindR Flask server | `http://localhost:8765/signals` |
| Symbol Filter | Sierra Chart symbol (ES, GC, NQ, etc.) | `ES` |
| Poll Interval | How often to check for signals (seconds) | `1.0` |
| Arrow Offset | Distance from price to draw arrows (ticks) | `5` |
| Show Position Line | Draw horizontal line at entry price | `Yes` |
| Max Signals | Maximum markers to display | `20` |
| Enable Alert Sound | Play sound on new signal | `Yes` |

## Signal Server Setup

### Option 1: Local Flask Server (Recommended for testing)

Run from your Pattern_FindR directory:

```bash
export SIERRA_BRIDGE_ENABLED=true
python -c "from sierra_chart_bridge import start_signal_server; start_signal_server(); import time; time.sleep(999999)"
```

This starts a server at `http://localhost:8765/signals`

### Option 2: Running with velocity_live_trader

The signal server can also run alongside velocity_live_trader:

```bash
export SIERRA_BRIDGE_ENABLED=true
export SIERRA_SIGNAL_FILE=~/pfr_signals.json
python velocity_live_trader.py
```

Signals are written to the JSON file. Run the Flask server separately to serve them.

### Option 3: ngrok for Remote Access

If Sierra Chart runs on a different machine:

```bash
ngrok http 8765
```

Use the ngrok URL (e.g., `https://abc123.ngrok.io/signals`) as your Signal Server URL.

## Troubleshooting

### "No signals appearing"

1. Check Signal Server URL is correct
2. Verify Symbol Filter matches your chart (ES, not SPY)
3. Confirm Flask server is running
4. Test in browser: `http://localhost:8765/signals`

### "HTTP request failed"

1. Firewall may be blocking the connection
2. Server may not be running
3. URL may be incorrect

### "Signals appearing on wrong bars"

The study draws signals at the most recent bar matching the price level.
For precise historical placement, the JSON would need bar timestamps.

## Symbol Mapping

Pattern_FindR uses equity symbols, Sierra Chart uses futures symbols:

| Pattern_FindR | Sierra Chart |
|---------------|--------------|
| SPY | ES |
| QQQ | NQ |
| IWM | RTY |
| GC=F | GC |
| ES=F | ES |
| BTC-USD | BTC.CME |

The `sierra_symbol` field in the JSON handles this mapping automatically.
