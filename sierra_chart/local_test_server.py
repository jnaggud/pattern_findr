#!/usr/bin/env python3
"""
Local Test Server for Sierra Chart Integration

Run this on your Windows machine to serve the sample_signals.json locally.
Sierra Chart's ACSIL study will poll this server.

Usage:
    python local_test_server.py

Then in Sierra Chart, set Signal Server URL to:
    http://localhost:8765/signals
"""

import http.server
import socketserver
import json
import os
from datetime import datetime

PORT = 8765
SIGNAL_FILE = "sample_signals.json"

class SignalHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/signals' or self.path == '/signals/':
            self.send_signals()
        elif self.path.startswith('/signals/'):
            # Filter by symbol
            symbol = self.path.split('/')[-1].upper()
            self.send_signals(symbol_filter=symbol)
        elif self.path == '/health':
            self.send_health()
        else:
            self.send_error(404, "Not Found. Try /signals or /health")

    def send_signals(self, symbol_filter=None):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            signal_path = os.path.join(script_dir, SIGNAL_FILE)

            with open(signal_path, 'r') as f:
                data = json.load(f)

            if symbol_filter:
                data['signals'] = [
                    s for s in data.get('signals', [])
                    if s.get('sierra_symbol') == symbol_filter
                ]
                data['active_positions'] = [
                    p for p in data.get('active_positions', [])
                    if p.get('sierra_symbol') == symbol_filter
                ]

            response = json.dumps(data, indent=2).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(response)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Served {len(data.get('signals', []))} signals")

        except FileNotFoundError:
            self.send_error(404, f"Signal file not found: {SIGNAL_FILE}")
        except json.JSONDecodeError as e:
            self.send_error(500, f"Invalid JSON: {e}")

    def send_health(self):
        response = json.dumps({
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        }).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    print("="*60)
    print("SIERRA CHART LOCAL TEST SERVER")
    print("="*60)
    print(f"\nServing signals on: http://localhost:{PORT}/signals")
    print(f"Signal file: {SIGNAL_FILE}")
    print("\nIn Sierra Chart, set Signal Server URL to:")
    print(f"    http://localhost:{PORT}/signals")
    print("\nEndpoints:")
    print(f"    http://localhost:{PORT}/signals      - All signals")
    print(f"    http://localhost:{PORT}/signals/ES   - ES signals only")
    print(f"    http://localhost:{PORT}/signals/GC   - GC signals only")
    print(f"    http://localhost:{PORT}/health       - Health check")
    print("\nPress Ctrl+C to stop")
    print("-"*60)

    with socketserver.TCPServer(("", PORT), SignalHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
