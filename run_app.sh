#!/bin/bash
# Startup script for Pattern_FindR with TensorFlow support

# Set environment variables to prevent TensorFlow crashes
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# Activate virtual environment and run the app
cd "$(dirname "$0")"
source venv-3.12/bin/activate
streamlit run app.py
