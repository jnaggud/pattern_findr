# Requirements Files Guide

This project uses multiple requirements files for different purposes:

## Files Overview

### `requirements.txt` - Main Requirements (Exact Versions)
- **Purpose**: Core dependencies with exact versions for production use
- **Usage**: `pip install -r requirements.txt`
- **When to use**: Fresh installations, production deployments, CI/CD pipelines

### `requirements-lock.txt` - Complete Lock File (pip freeze)
- **Purpose**: ALL installed packages with exact versions from `pip freeze`
- **Usage**: `pip install -r requirements-lock.txt`  
- **When to use**: Exact environment reproduction, debugging conflicts

### `environment.yml` - Conda Environment Export (Most Comprehensive)
- **Purpose**: Complete conda environment with conda + pip packages, channels, dependencies
- **Usage**: `conda env create -f environment.yml`
- **When to use**: **RECOMMENDED** for full environment reproduction (530+ packages)

### `requirements.lock.txt` - Legacy Full Environment
- **Purpose**: Old comprehensive freeze (can be removed)
- **Status**: Replaced by requirements-lock.txt

### `requirements_ml.txt` - ML Extensions
- **Purpose**: Additional ML packages with minimum version requirements
- **Usage**: `pip install -r requirements_ml.txt` (after main requirements)
- **When to use**: When you need extra ML capabilities

## Quick Start

For new installations:
```bash
# Option 1: Create complete conda environment (RECOMMENDED)
conda env create -f environment.yml
conda activate pattern-findr

# Option 2: Install core requirements only (lightweight)
pip install -r requirements.txt

# Option 3: Install all pip packages (exact reproduction)
pip install -r requirements-lock.txt

# Optional: Add extra ML packages
pip install -r requirements_ml.txt
```

## Generating Lock Files

### Simple Commands (Recommended)
```bash
# Generate conda environment export (most comprehensive)
conda env export > environment.yml

# Generate pip freeze lock file
pip freeze > requirements-lock.txt
```

### Using Scripts
```bash
# Option 1: Bash script
./update_lock.sh

# Option 2: Python script  
python generate_lock.py
```

## Version Management

- **requirements.txt**: Hand-curated core packages with exact versions
- **environment.yml**: Complete conda environment (conda + pip packages, channels)
- **requirements-lock.txt**: Complete `pip freeze` output for exact reproduction
- **Recommended**: Use `environment.yml` for most comprehensive reproduction

## Platform Notes

- `tensorflow-metal==1.2.0`: macOS GPU acceleration (Apple Silicon)
- For other platforms, you may need to adjust TensorFlow installation
- Conda environment includes platform-specific builds automatically

## Updating Dependencies

After adding/updating packages:

1. Install and test new packages
2. Regenerate environment files:
   ```bash
   conda env export > environment.yml
   pip freeze > requirements-lock.txt
   ```
3. Update requirements.txt if needed
4. Test installation in clean environment:
   ```bash
   conda env create -f environment.yml -n test-env
   ```
5. Commit changes

## Troubleshooting

- If you get dependency conflicts, try using the full `requirements.lock.txt`
- For Apple Silicon Macs, ensure `tensorflow-metal` is installed for GPU support
- If packages are missing, check that you're using the correct Python environment
