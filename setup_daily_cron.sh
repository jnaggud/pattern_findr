#!/bin/bash
# =============================================================================
# DAILY DATA PIPELINE - CRON JOB SETUP
# =============================================================================
#
# This script sets up automated daily data collection for:
# 1. IV (Implied Volatility) data from Polygon
# 2. News sentiment from Finnhub or free scraper
# 3. Price data synchronization
#
# All data is saved to SQLite databases for use by the prediction models.
#
# Usage:
#   chmod +x setup_daily_cron.sh
#   ./setup_daily_cron.sh
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH=$(which python3 || which python)
LOG_DIR="${SCRIPT_DIR}/logs"

echo "=============================================="
echo "DAILY DATA PIPELINE - CRON SETUP"
echo "=============================================="
echo "Script Directory: ${SCRIPT_DIR}"
echo "Python Path: ${PYTHON_PATH}"
echo ""

# Create logs directory
mkdir -p "${LOG_DIR}"
echo "Created logs directory: ${LOG_DIR}"

# Create the daily runner script
RUNNER_SCRIPT="${SCRIPT_DIR}/run_daily_pipeline.sh"

cat > "${RUNNER_SCRIPT}" << 'RUNNER_EOF'
#!/bin/bash
# Daily Pipeline Runner - Called by cron

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
DATE=$(date +%Y%m%d)

# Activate virtual environment if it exists
if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "========================================" >> "${LOG_DIR}/pipeline_${DATE}.log"
echo "Starting Daily Pipeline: $(date)" >> "${LOG_DIR}/pipeline_${DATE}.log"
echo "========================================" >> "${LOG_DIR}/pipeline_${DATE}.log"

# Run the comprehensive daily pipeline
cd "${SCRIPT_DIR}"
python3 daily_data_pipeline.py >> "${LOG_DIR}/pipeline_${DATE}.log" 2>&1

# Also run standalone IV collection (backup)
python3 collect_iv_data.py >> "${LOG_DIR}/iv_${DATE}.log" 2>&1

echo "Pipeline completed: $(date)" >> "${LOG_DIR}/pipeline_${DATE}.log"

# Clean up old logs (keep last 30 days)
find "${LOG_DIR}" -name "*.log" -mtime +30 -delete
RUNNER_EOF

chmod +x "${RUNNER_SCRIPT}"
echo "Created runner script: ${RUNNER_SCRIPT}"

# Show current crontab
echo ""
echo "Current crontab entries:"
crontab -l 2>/dev/null || echo "(empty)"

# Generate cron entry
# Run at 5:00 PM EST (17:00) on weekdays (Mon-Fri)
# Adjust time based on your timezone
CRON_ENTRY="0 17 * * 1-5 ${RUNNER_SCRIPT}"

echo ""
echo "=============================================="
echo "RECOMMENDED CRON ENTRY"
echo "=============================================="
echo ""
echo "To collect data after market close (5 PM EST weekdays):"
echo ""
echo "  ${CRON_ENTRY}"
echo ""
echo "To add this to your crontab, run:"
echo ""
echo "  crontab -e"
echo ""
echo "Then add the line above to the file."
echo ""
echo "Or run this command to add it automatically:"
echo ""
echo "  (crontab -l 2>/dev/null; echo '${CRON_ENTRY}') | crontab -"
echo ""

# Ask user if they want to install
read -p "Would you like to install this cron job now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    (crontab -l 2>/dev/null | grep -v "run_daily_pipeline.sh"; echo "${CRON_ENTRY}") | crontab -
    echo "Cron job installed successfully!"
    echo ""
    echo "Your new crontab:"
    crontab -l
else
    echo "Skipped. You can install manually later."
fi

echo ""
echo "=============================================="
echo "MANUAL TESTING"
echo "=============================================="
echo ""
echo "To test the pipeline manually:"
echo ""
echo "  # Quick test (SPY only)"
echo "  python3 ${SCRIPT_DIR}/daily_data_pipeline.py --test"
echo ""
echo "  # Full run with default tickers"
echo "  python3 ${SCRIPT_DIR}/daily_data_pipeline.py"
echo ""
echo "  # Check logs"
echo "  tail -f ${LOG_DIR}/pipeline_$(date +%Y%m%d).log"
echo ""
echo "=============================================="
echo "SETUP COMPLETE"
echo "=============================================="
