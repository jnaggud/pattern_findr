#!/bin/bash
# Startup script for Pattern_FindR using conda environment

cd "$(dirname "$0")"

# Use the conda environment's Python directly
/opt/anaconda3/envs/pattern_findr/bin/streamlit run app.py
