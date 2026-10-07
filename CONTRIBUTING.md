# Contributing

Use GitHub Issues for reproducible bugs and focused feature proposals. Include the Python version, operating system, relevant command or dashboard page, and a minimal example. Remove credentials, webhook URLs, and private account information from logs and attachments.

For code changes:

1. Create a branch and keep the change focused.
2. Follow the existing style in the affected module.
3. Install the core and test requirements, then run `python -m pytest tests/unit -q`.
4. Run relevant integration or regression tests when changing data handling, backtests, or state transitions.
5. Explain the behavior change and validation in the pull request.

Use synthetic or publicly shareable data for examples. Do not commit virtual environments, local strategy state, trained models, API keys, or notification endpoints. Performance-related changes should describe the evaluation period, signal timing, costs, and out-of-sample methodology.
