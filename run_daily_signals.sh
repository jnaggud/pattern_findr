#!/bin/bash
# Daily Signal Generator Runner
# This script runs the signal generator and logs output

# Navigate to project directory
cd "$(dirname "$0")"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pattern_findr

# Create logs directory if it doesn't exist
mkdir -p logs

# Run signal generator and log output
LOGFILE="logs/signals_$(date +%Y-%m-%d_%H%M%S).log"
echo "🚀 Starting daily signal scan at $(date)" | tee "$LOGFILE"
echo "================================================" | tee -a "$LOGFILE"

/opt/anaconda3/envs/pattern_findr/bin/python daily_signal_generator.py 2>&1 | tee -a "$LOGFILE"

echo "================================================" | tee -a "$LOGFILE"
echo "✅ Daily signal scan completed at $(date)" | tee -a "$LOGFILE"
echo "📄 Log saved to: $LOGFILE" | tee -a "$LOGFILE"

# Optional: Display the latest signals
if [ -f "daily_signals/signals_$(date +%Y-%m-%d).csv" ]; then
    echo ""
    echo "📊 Today's signals saved to: daily_signals/signals_$(date +%Y-%m-%d).csv"
fi
