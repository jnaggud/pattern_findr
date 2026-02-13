"""
Sierra Chart Bridge - Signal Publishing Module

Publishes Pattern_FindR trading signals to a JSON file that can be read by
Sierra Chart's ACSIL custom study or served via HTTP endpoint.

Usage:
    from sierra_chart_bridge import publish_signal_to_sierra, build_sierra_signal

    signal = build_sierra_signal("ENTRY", "LONG", "SPY", 595.42, signal_time,
                                  strategy_name, config)
    publish_signal_to_sierra(signal, "/path/to/signals.json")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd

# Symbol mapping: Pattern_FindR ticker -> Sierra Chart symbol
SYMBOL_MAP = {
    "SPY": "ES",
    "QQQ": "NQ",
    "IWM": "RTY",
    "BTC-USD": "BTC.CME",
    "GC=F": "GC",
    "ES=F": "ES",
    "NQ=F": "NQ",
    "RTY=F": "RTY",
    "CL=F": "CL",
    "SI=F": "SI",
    "^GSPC": "ES",
    "^NDX": "NQ",
}

# Default signal file path
DEFAULT_SIGNAL_FILE = os.path.expanduser("~/pfr_signals.json")


def normalize_timestamp(ts: Any) -> str:
    """
    Normalize any timestamp to consistent ISO 8601 format with UTC timezone.

    Handles: datetime, pd.Timestamp, str (various formats), int/float (Unix time)
    Returns: ISO 8601 format string with Z suffix (YYYY-MM-DDTHH:MM:SSZ)

    All timestamps are assumed to be UTC (futures data from yfinance is UTC).
    """
    def ensure_utc_suffix(iso_str: str) -> str:
        """Add Z suffix if not already present."""
        # Remove microseconds for cleaner output
        if '.' in iso_str:
            iso_str = iso_str.split('.')[0]
        # Add Z if no timezone indicator
        if not iso_str.endswith('Z') and '+' not in iso_str and '-' not in iso_str[-6:]:
            iso_str += 'Z'
        return iso_str

    if ts is None:
        from datetime import timezone
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Already a datetime
    if isinstance(ts, datetime):
        return ensure_utc_suffix(ts.isoformat())

    # Pandas Timestamp
    if isinstance(ts, pd.Timestamp):
        return ensure_utc_suffix(ts.isoformat())

    # Unix timestamp (int or float) - UTC by definition
    if isinstance(ts, (int, float)):
        from datetime import timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # String - try to parse and normalize
    if isinstance(ts, str):
        # Already has timezone indicator
        if ts.endswith('Z') or '+' in ts or (len(ts) > 6 and ts[-6] == '-' and ':' in ts[-5:]):
            return ts

        # Already ISO format with T separator - just add Z
        if 'T' in ts and '-' in ts:
            return ensure_utc_suffix(ts)

        # Common formats to try
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(ts, fmt)
                return ensure_utc_suffix(parsed.isoformat())
            except ValueError:
                continue

        # Fallback: replace space with T if it looks like a date
        if ' ' in ts and '-' in ts:
            return ensure_utc_suffix(ts.replace(' ', 'T', 1))

    # Last resort: just convert to string
    return str(ts)


def load_symbol_map(config_path: str = None) -> Dict[str, str]:
    """Load symbol mapping from JSON config file, falling back to defaults."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                custom_map = json.load(f)
                return {**SYMBOL_MAP, **custom_map}
        except Exception as e:
            print(f"Warning: Could not load symbol map from {config_path}: {e}")
    return SYMBOL_MAP


def get_sierra_symbol(ticker: str, symbol_map: Dict[str, str] = None) -> str:
    """Convert Pattern_FindR ticker to Sierra Chart symbol."""
    if symbol_map is None:
        symbol_map = SYMBOL_MAP
    return symbol_map.get(ticker, ticker)


def build_sierra_signal(
    signal_type: str,
    direction: str,
    ticker: str,
    price: float,
    signal_time: Any,
    strategy_name: str,
    config: Dict[str, Any],
    exit_data: Optional[Dict[str, Any]] = None,
    symbol_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Build a signal dict in Sierra Chart-compatible format.

    Args:
        signal_type: "ENTRY" or "EXIT"
        direction: "LONG" or "SHORT"
        ticker: Pattern_FindR ticker (e.g., "SPY", "GC=F")
        price: Signal price
        signal_time: Datetime of the signal
        strategy_name: Name of the strategy
        config: Strategy configuration dict
        exit_data: Optional dict with exit-specific data (entry_price, pnl_pct, reason)
        symbol_map: Optional custom symbol mapping

    Returns:
        Signal dict ready for JSON serialization
    """
    if symbol_map is None:
        symbol_map = SYMBOL_MAP

    signal = {
        "id": f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "signal_type": signal_type,
        "direction": direction,
        "ticker": ticker,
        "sierra_symbol": get_sierra_symbol(ticker, symbol_map),
        "signal_time": normalize_timestamp(signal_time),
        "price": float(price),
        "strategy_name": strategy_name,
    }

    # Add risk levels for entry signals
    if signal_type == "ENTRY":
        stop_loss_pct = config.get('stop_loss_pct', 0)
        take_profit_pct = config.get('take_profit_pct', 0)

        if direction == "LONG":
            signal["stop_loss"] = round(price * (1 - stop_loss_pct / 100), 2)
            signal["take_profit"] = round(price * (1 + take_profit_pct / 100), 2)
        else:  # SHORT
            signal["stop_loss"] = round(price * (1 + stop_loss_pct / 100), 2)
            signal["take_profit"] = round(price * (1 - take_profit_pct / 100), 2)

    # Add exit-specific data
    if exit_data:
        signal.update(exit_data)

    return signal


def publish_signal_to_sierra(
    signal: Dict[str, Any],
    file_path: str = None,
    max_signals: int = 50
) -> bool:
    """
    Atomically write signal to JSON file.

    Uses write-to-temp-then-rename pattern to prevent partial reads
    by Sierra Chart while we're writing.

    Args:
        signal: Signal dict from build_sierra_signal()
        file_path: Path to signal file (default: ~/pfr_signals.json)
        max_signals: Maximum signals to keep in history (default: 50)

    Returns:
        True if successful, False otherwise
    """
    if file_path is None:
        file_path = DEFAULT_SIGNAL_FILE

    path = Path(file_path)

    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing signals
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted file, start fresh
                data = {"version": "1.0", "signals": [], "active_positions": []}
        else:
            data = {"version": "1.0", "signals": [], "active_positions": []}

        # Add new signal, keep last N
        data["signals"].append(signal)
        data["signals"] = data["signals"][-max_signals:]
        data["last_updated"] = datetime.now().isoformat()

        # Update active positions
        ticker = signal.get("ticker")
        if signal["signal_type"] == "ENTRY":
            # Remove any existing position for this ticker first
            data["active_positions"] = [
                p for p in data.get("active_positions", [])
                if p.get("ticker") != ticker
            ]
            # Add new position
            data["active_positions"].append({
                "ticker": ticker,
                "sierra_symbol": signal.get("sierra_symbol", ticker),
                "direction": signal["direction"],
                "entry_price": signal["price"],
                "entry_time": normalize_timestamp(signal["signal_time"]),
                "strategy_name": signal.get("strategy_name", "")
            })
        elif signal["signal_type"] == "EXIT":
            # Remove position for this ticker
            data["active_positions"] = [
                p for p in data.get("active_positions", [])
                if p.get("ticker") != ticker
            ]

        # Atomic write: write to temp file, then rename
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Atomic rename (works on POSIX and modern Windows)
        temp_path.replace(path)

        return True

    except Exception as e:
        print(f"Error publishing signal to Sierra Chart: {e}")
        return False


def get_active_positions(file_path: str = None) -> list:
    """Read current active positions from signal file."""
    if file_path is None:
        file_path = DEFAULT_SIGNAL_FILE

    path = Path(file_path)
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("active_positions", [])
    except Exception:
        return []


def clear_signals(file_path: str = None) -> bool:
    """Clear all signals and positions from the file."""
    if file_path is None:
        file_path = DEFAULT_SIGNAL_FILE

    path = Path(file_path)

    try:
        data = {
            "version": "1.0",
            "signals": [],
            "active_positions": [],
            "last_updated": datetime.now().isoformat(),
            "cleared_at": datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(f"Error clearing signals: {e}")
        return False


# Optional: Simple HTTP server for serving signals
def start_signal_server(
    signal_file: str = None,
    host: str = "0.0.0.0",
    port: int = 8765
):
    """
    Start a simple HTTP server to serve signals.

    Sierra Chart can poll this endpoint using sc.HTTPRequest().

    Args:
        signal_file: Path to signal JSON file
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to listen on (default: 8765)
    """
    try:
        from flask import Flask, jsonify, send_file
        import threading
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return None

    if signal_file is None:
        signal_file = DEFAULT_SIGNAL_FILE

    app = Flask(__name__)

    @app.route('/signals')
    def get_signals():
        """Return all signals as JSON."""
        try:
            return send_file(signal_file, mimetype='application/json')
        except FileNotFoundError:
            return jsonify({"version": "1.0", "signals": [], "active_positions": []})

    @app.route('/signals/<ticker>')
    def get_signals_by_ticker(ticker: str):
        """Return signals filtered by ticker."""
        try:
            with open(signal_file) as f:
                data = json.load(f)

            # Filter by ticker or sierra_symbol
            filtered = [
                s for s in data.get("signals", [])
                if s.get("ticker") == ticker or s.get("sierra_symbol") == ticker
            ]

            active = [
                p for p in data.get("active_positions", [])
                if p.get("ticker") == ticker or p.get("sierra_symbol") == ticker
            ]

            return jsonify({
                "version": data.get("version", "1.0"),
                "signals": filtered,
                "active_positions": active,
                "last_updated": data.get("last_updated")
            })
        except FileNotFoundError:
            return jsonify({"version": "1.0", "signals": [], "active_positions": []})

    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    # Run in background thread
    def run_server():
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print(f"Sierra Chart signal server started at http://{host}:{port}/signals")
    return server_thread


if __name__ == "__main__":
    # Test the module
    print("Testing Sierra Chart Bridge...")

    # Create a test signal
    test_config = {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0
    }

    signal = build_sierra_signal(
        signal_type="ENTRY",
        direction="LONG",
        ticker="SPY",
        price=595.42,
        signal_time=datetime.now(),
        strategy_name="velocity_SPY_test",
        config=test_config
    )

    print(f"Built signal: {json.dumps(signal, indent=2)}")

    # Publish to test file
    test_file = "/tmp/pfr_signals_test.json"
    success = publish_signal_to_sierra(signal, test_file)
    print(f"Published to {test_file}: {success}")

    # Read back
    if success:
        with open(test_file) as f:
            data = json.load(f)
        print(f"Signal file contents: {json.dumps(data, indent=2)}")
