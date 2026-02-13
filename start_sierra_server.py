#!/usr/bin/env python3
"""
Sierra Chart Signal Server

Run this in a separate terminal to serve signals to Sierra Chart subscribers.

Usage:
    python start_sierra_server.py

Then expose via ngrok in another terminal:
    ngrok http 8765
"""

import os
import sys
import time

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sierra_chart_bridge import start_signal_server

# Configuration
SIGNAL_FILE = os.path.expanduser("~/pfr_signals.json")
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8765

def main():
    print("=" * 60)
    print("  SIERRA CHART SIGNAL SERVER")
    print("=" * 60)
    print()
    print(f"Signal File: {SIGNAL_FILE}")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print()
    print("Endpoints:")
    print(f"  http://localhost:{PORT}/signals      - All signals")
    print(f"  http://localhost:{PORT}/signals/ES   - ES signals only")
    print(f"  http://localhost:{PORT}/signals/GC   - GC signals only")
    print(f"  http://localhost:{PORT}/health       - Health check")
    print()
    print("To expose to internet, run in another terminal:")
    print(f"  ngrok http {PORT}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Start the server (runs in background thread)
    start_signal_server(signal_file=SIGNAL_FILE, host=HOST, port=PORT)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
