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
