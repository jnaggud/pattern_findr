#!/bin/bash
# Startup script for Pattern_FindR using conda environment

cd "$(dirname "$0")"

# Use the conda environment's Python directly
/Users/jeffersonduggan/miniconda3/envs/pattern_findr_312/bin/streamlit run app.py
