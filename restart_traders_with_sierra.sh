#!/bin/bash
# Restart legacy traders with Sierra Chart bridge enabled
#
# Usage: ./restart_traders_with_sierra.sh
#
# This will:
# 1. Show currently running traders
# 2. Ask for confirmation before stopping each one
# 3. Restart with SIERRA_BRIDGE_ENABLED=true

cd /Users/jeffersonduggan/Documents/Pattern_FindR

export SIERRA_BRIDGE_ENABLED=true
export SIERRA_SIGNAL_FILE=~/pfr_signals.json

echo "=============================================="
echo "  RESTART TRADERS WITH SIERRA CHART ENABLED"
echo "=============================================="
echo ""
echo "Environment variables that will be set:"
echo "  SIERRA_BRIDGE_ENABLED=true"
echo "  SIERRA_SIGNAL_FILE=~/pfr_signals.json"
echo ""

# Find velocity_live_trader.py processes
echo "Currently running legacy traders:"
echo ""
ps aux | grep "velocity_live_trader.py" | grep -v grep | awk '{print "  PID: "$2"  TTY: "$7"  Started: "$9}'
echo ""

echo "To restart a trader in a specific terminal:"
echo ""
echo "  1. Go to that terminal window"
echo "  2. Press Ctrl+C to stop the trader"
echo "  3. Run this command:"
echo ""
echo "     export SIERRA_BRIDGE_ENABLED=true && python velocity_live_trader.py"
echo ""
echo "Or with a specific config:"
echo ""
echo "     export SIERRA_BRIDGE_ENABLED=true && python velocity_live_trader.py --config velocity_strategies/velocity_ES=F_15m_v5/velocity_config.json"
echo ""
echo "=============================================="
