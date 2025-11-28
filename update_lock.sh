#!/bin/bash
# Generate lock files for both pip and conda environments
# This captures ALL installed packages with exact versions

echo "🔄 Generating lock files..."

# 1. Generate pip freeze lock file
echo "📦 Creating requirements-lock.txt (pip freeze)..."
echo "# Generated on M3 Mac Studio - $(date +%Y-%m-%d)" > requirements-lock.txt
echo "# This is the environment that trained the production models" >> requirements-lock.txt
echo "# Pattern_FindR - Complete Lock File (pip freeze)" >> requirements-lock.txt
echo "# Use this for exact environment reproduction" >> requirements-lock.txt  
echo "# Install with: pip install -r requirements-lock.txt" >> requirements-lock.txt
echo "" >> requirements-lock.txt
pip freeze >> requirements-lock.txt

# 2. Generate conda environment export (most comprehensive)
echo "🐍 Creating environment.yml (conda env export)..."
conda env export > environment.yml

# Add hardware context to environment.yml
temp_file=$(mktemp)
echo "# Generated on M3 Mac Studio - $(date +%Y-%m-%d)" > "$temp_file"
echo "# This is the environment that trained the production models" >> "$temp_file"
echo "# Complete conda environment export (conda + pip packages)" >> "$temp_file"
echo "# Install with: conda env create -f environment.yml" >> "$temp_file"
echo "" >> "$temp_file"
tail -n +2 environment.yml >> "$temp_file"  # Skip original first line (name: pattern-findr)
echo "name: pattern-findr" > environment.yml
cat "$temp_file" >> environment.yml
rm "$temp_file"

# Report results
pip_count=$(pip freeze | wc -l | tr -d ' ')
conda_lines=$(wc -l < environment.yml | tr -d ' ')

echo "✅ Generated lock files:"
echo "   📄 requirements-lock.txt: $pip_count pip packages"
echo "   📄 environment.yml: $conda_lines total lines (conda + pip)"
echo ""
echo "💡 For new environments:"
echo "   conda env create -f environment.yml"
echo "   # OR"  
echo "   pip install -r requirements-lock.txt"
