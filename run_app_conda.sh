#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PATTERN_FINDR_PYTHON:-python}"
if ! "$PYTHON_BIN" -c 'import streamlit' >/dev/null 2>&1; then
    printf '%s\n' 'Activate your Python 3.10 environment and install requirements.txt first.' >&2
    exit 1
fi
exec "$PYTHON_BIN" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true "$@"
