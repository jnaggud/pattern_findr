"""Exercise the dashboard with a deterministic Yahoo-style response."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("index_name", ["Date", "Datetime"])
def test_find_patterns_accepts_yahoo_multiindex(monkeypatch, tmp_path, index_name):
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    monkeypatch.chdir(tmp_path)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]],
        names=["Price", "Ticker"],
    )
    prices = pd.DataFrame(
        [[100, 103, 99, 102, 1000], [102, 105, 101, 104, 1200],
         [104, 106, 102, 103, 1100], [103, 108, 102, 107, 1500]],
        index=pd.date_range("2025-01-02", periods=4, name=index_name),
        columns=columns,
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: prices.copy())
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not app.exception
    button = next(b for b in app.sidebar.button if b.label == "Find Patterns")
    button.click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) > 0
