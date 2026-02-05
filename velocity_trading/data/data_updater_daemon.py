#!/usr/bin/env python3
"""
Data Updater Daemon - Keeps OHLCV data current independently of traders.

This daemon:
1. Auto-discovers active strategies from velocity_strategies/*/velocity_config.json
2. Updates data for all unique ticker/interval pairs continuously
3. Respects market hours (more frequent when open, sleeps when closed)
4. Writes to the same SQLite databases used by traders (DataPipeline)
5. Runs independently via launchd (macOS) or manually

Usage:
    # Run directly
    python -m velocity_trading.data.data_updater_daemon

    # Run with specific config
    python -m velocity_trading.data.data_updater_daemon --interval 60 --verbose

    # List discovered strategies (dry run)
    python -m velocity_trading.data.data_updater_daemon --discover-only

Install as launchd service:
    cp com.patternfindr.data-updater.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.patternfindr.data-updater.plist
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from velocity_trading.data.data_pipeline import DataPipeline

logger = logging.getLogger('data_updater')

# Strategy directory
STRATEGIES_DIR = PROJECT_ROOT / 'velocity_strategies'

# Health check file
HEALTH_FILE = Path('/tmp/patternfindr_data_updater_health.json')


class DataUpdaterDaemon:
    """
    Standalone daemon that keeps OHLCV data updated for all active strategies.

    Discovers strategies from velocity_strategies/*/velocity_config.json,
    extracts unique ticker/interval pairs, and runs DataPipeline.update()
    on each at appropriate intervals.
    """

    def __init__(self, update_interval: int = 30, verbose: bool = False):
        """
        Args:
            update_interval: Seconds between update cycles (default: 30)
            verbose: Print detailed status messages
        """
        self.update_interval = update_interval
        self.verbose = verbose
        self._running = False
        self._pipelines: Dict[Tuple[str, str], DataPipeline] = {}
        self._error_counts: Dict[Tuple[str, str], int] = {}
        self._last_update_times: Dict[Tuple[str, str], float] = {}
        self._cycle_count = 0

    def discover_strategies(self) -> List[Dict]:
        """
        Discover active strategies from velocity_strategies directory.

        Returns:
            List of dicts with keys: strategy_name, ticker, interval, config_path
        """
        strategies = []

        if not STRATEGIES_DIR.exists():
            logger.warning(f"Strategies directory not found: {STRATEGIES_DIR}")
            return strategies

        for config_path in sorted(STRATEGIES_DIR.glob('*/velocity_config.json')):
            try:
                with open(config_path) as f:
                    config = json.load(f)

                ticker = config.get('ticker')
                interval = config.get('interval')
                strategy_name = config.get('strategy_name', config_path.parent.name)

                if not ticker or not interval:
                    continue

                strategies.append({
                    'strategy_name': strategy_name,
                    'ticker': ticker,
                    'interval': interval,
                    'config_path': str(config_path),
                })

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read {config_path}: {e}")

        return strategies

    def get_unique_pairs(self, strategies: List[Dict]) -> Set[Tuple[str, str]]:
        """Extract unique ticker/interval pairs from strategies."""
        return {(s['ticker'], s['interval']) for s in strategies}

    def _init_pipelines(self, pairs: Set[Tuple[str, str]]):
        """Initialize DataPipeline instances for each pair."""
        for ticker, interval in pairs:
            key = (ticker, interval)
            if key not in self._pipelines:
                try:
                    self._pipelines[key] = DataPipeline(ticker, interval)
                    self._error_counts[key] = 0
                    logger.info(f"Initialized pipeline for {ticker}/{interval}")
                except Exception as e:
                    logger.error(f"Failed to initialize pipeline for {ticker}/{interval}: {e}")

    def _should_update(self, key: Tuple[str, str]) -> bool:
        """
        Check if a ticker/interval pair needs updating.

        Considers:
        - Time since last update
        - Market hours for the ticker
        - Error count (back off on repeated failures)
        """
        ticker, interval = key
        now = time.time()
        last_update = self._last_update_times.get(key, 0)
        elapsed = now - last_update

        # Back off on repeated errors
        errors = self._error_counts.get(key, 0)
        if errors > 0:
            backoff = min(self.update_interval * (2 ** min(errors, 5)), 600)
            if elapsed < backoff:
                return False

        # Minimum interval between updates based on bar interval
        min_intervals = {
            '1m': 15, '5m': 30, '15m': 30,
            '30m': 60, '1h': 120, '1d': 300,
        }
        min_interval = min_intervals.get(interval, 30)

        if elapsed < min_interval:
            return False

        # Check market hours
        try:
            from velocity_trading.data.market_hours import is_market_open
            if not is_market_open(ticker):
                # When market is closed, update much less frequently
                if elapsed < 300:  # At most every 5 minutes when closed
                    return False
        except ImportError:
            pass  # Market hours module not available, always update

        return True

    def _update_one(self, key: Tuple[str, str]):
        """Update a single ticker/interval pair."""
        ticker, interval = key
        pipeline = self._pipelines.get(key)
        if not pipeline:
            return

        try:
            success, result = pipeline.update()
            self._last_update_times[key] = time.time()

            if success:
                self._error_counts[key] = 0
                new_bars = result.get('new_bars', 0)
                if new_bars > 0 and self.verbose:
                    logger.info(f"Updated {ticker}/{interval}: +{new_bars} bars")
            else:
                self._error_counts[key] = self._error_counts.get(key, 0) + 1
                logger.warning(f"Update failed for {ticker}/{interval}: {result}")

        except Exception as e:
            self._error_counts[key] = self._error_counts.get(key, 0) + 1
            logger.error(f"Error updating {ticker}/{interval}: {e}")

    def _write_health_status(self):
        """Write health status for external monitoring."""
        try:
            status = {
                'last_update': datetime.now().isoformat(),
                'cycle_count': self._cycle_count,
                'pipelines_active': len(self._pipelines),
                'errors': {f"{t}/{i}": c for (t, i), c in self._error_counts.items() if c > 0},
                'pid': os.getpid(),
            }
            with open(HEALTH_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        except OSError:
            pass  # Non-critical

    def run_once(self):
        """Run a single update cycle across all pairs."""
        for key in list(self._pipelines.keys()):
            if not self._running:
                break
            if self._should_update(key):
                self._update_one(key)

        self._cycle_count += 1
        if self._cycle_count % 10 == 0:
            self._write_health_status()

    def run_forever(self):
        """Run update loop until stopped."""
        self._running = True

        # Discover strategies
        strategies = self.discover_strategies()
        pairs = self.get_unique_pairs(strategies)

        if not pairs:
            logger.error("No ticker/interval pairs found. Check velocity_strategies/ directory.")
            return

        logger.info(f"Data updater starting: {len(pairs)} unique ticker/interval pairs")
        for ticker, interval in sorted(pairs):
            strats = [s['strategy_name'] for s in strategies if s['ticker'] == ticker and s['interval'] == interval]
            logger.info(f"  {ticker}/{interval} -> used by: {', '.join(strats)}")

        self._init_pipelines(pairs)

        # Signal handlers
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, stopping...")
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        # Main loop
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Sleep in small increments for responsive shutdown
            sleep_remaining = self.update_interval
            while sleep_remaining > 0 and self._running:
                time.sleep(min(sleep_remaining, 1.0))
                sleep_remaining -= 1.0

        logger.info("Data updater stopped.")
        self._write_health_status()

    def stop(self):
        """Stop the daemon gracefully."""
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Data Updater Daemon")
    parser.add_argument('--interval', type=int, default=30,
                        help="Seconds between update cycles (default: 30)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Print detailed status messages")
    parser.add_argument('--discover-only', action='store_true',
                        help="List discovered strategies and exit")
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    daemon = DataUpdaterDaemon(
        update_interval=args.interval,
        verbose=args.verbose,
    )

    if args.discover_only:
        strategies = daemon.discover_strategies()
        pairs = daemon.get_unique_pairs(strategies)

        print(f"\nDiscovered {len(strategies)} strategies with {len(pairs)} unique ticker/interval pairs:\n")
        for ticker, interval in sorted(pairs):
            strats = [s['strategy_name'] for s in strategies if s['ticker'] == ticker and s['interval'] == interval]
            print(f"  {ticker:12s} {interval:4s} -> {len(strats)} strategies: {', '.join(strats[:3])}")
            if len(strats) > 3:
                print(f"{'':21s} ... and {len(strats) - 3} more")
        print()
        return

    daemon.run_forever()


if __name__ == '__main__':
    main()
