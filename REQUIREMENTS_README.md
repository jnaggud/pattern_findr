# Installation and dependencies

## Supported starting point

Use Python 3.10 in a dedicated virtual environment or Conda environment. The primary development platform is Apple Silicon macOS. TensorFlow is configured for CPU execution by the main application; `tensorflow-metal` is not required for this workflow.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_app_conda.sh
```

If Python 3.10 is managed by Conda:

```bash
conda create -n pattern-findr python=3.10 -y
conda activate pattern-findr
python -m pip install -r requirements.txt
./run_app_conda.sh
```

Open http://localhost:8501 in Safari on macOS. The launcher binds to the local machine and does not open a browser automatically. To choose an interpreter explicitly, set `PATTERN_FINDR_PYTHON` to its path. Additional Streamlit arguments can be passed to the launcher.

## Dependency files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Core dashboard, indicators, optimization, and model libraries |
| `requirements-test.txt` | Test runner, installed after the core environment |
| `requirements_ml.txt` | Historical optional ML package list; assess compatibility before using it |
| `environment.yml`, `requirements-lock.txt`, `requirements.lock.txt` | Historical development-environment exports, retained for reference rather than a portable installation recipe |

The repository includes pandas-ta 0.3.14b0 and its distribution metadata because existing indicator code relies on that API. Run the application from the repository root. The setuptools pin preserves the `pkg_resources` API used by that bundled version. Do not install a newer pandas-ta release over it without testing compatibility.

## Optional features

- Databento and Polygon integrations require their respective packages, credentials, and data entitlements.
- Some experimental pages use LightGBM, imbalanced-learn, MAPIE, or other optional libraries. These are not part of the core quick start. In particular, the legacy conformal helper imports `MapieClassifier`; newer MAPIE APIs are not a drop-in replacement.
- Optional language-model and SMS integrations require their own packages and credentials.
- Model weights are generated locally and are not included in the repository. The chart-pattern training workflow is available from the dashboard.
- Existing chart-training images are stored with Git LFS. Install Git LFS and run `git lfs pull` when those assets are needed.

## Verification

```bash
python -m pip check
python -m pip install -r requirements-test.txt
python -m pytest tests/unit -q
```

Run `tests/` explicitly: several root-level files with test-like names are research scripts that make network requests or run long experiments.

Keep credentials in ignored local configuration or environment variables; see [configuration](docs/configuration.md).
