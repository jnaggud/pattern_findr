"""
Structured logging configuration for velocity trading system.

Provides JSON-formatted logging for machine-parseable logs alongside
human-readable console output.

Usage:
    from velocity_trading.logging_config import setup_logging
    setup_logging(verbose=True)

    # Then in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Message", extra={'trade_id': 'abc123'})
"""

import json
import logging
import sys
from datetime import datetime
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log entries for easy parsing and monitoring."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add optional structured fields if present
        for field in ('trade_id', 'strategy', 'ticker', 'interval',
                      'action', 'price', 'signal', 'error_type',
                      'cycle_count', 'new_bars'):
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        # Include exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname[0]  # Single char: I, W, E, D
        name = record.name.split('.')[-1]  # Short module name
        msg = record.getMessage()

        # Add context fields inline if present
        strategy = getattr(record, 'strategy', None)
        if strategy:
            return f"{ts} [{level}] [{strategy}] {msg}"
        return f"{ts} [{level}] [{name}] {msg}"


def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None,
    json_log_file: Optional[str] = None,
):
    """
    Configure logging for the velocity trading system.

    Args:
        verbose: If True, set console to DEBUG level
        log_file: Optional path for human-readable log file
        json_log_file: Optional path for JSON-structured log file
    """
    root_logger = logging.getLogger('velocity_trading')
    root_logger.setLevel(logging.DEBUG)

    # Also configure the data_updater logger
    updater_logger = logging.getLogger('data_updater')
    updater_logger.setLevel(logging.DEBUG)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)
    updater_logger.addHandler(console_handler)

    # JSON file handler (structured, for monitoring)
    if json_log_file:
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)
        updater_logger.addHandler(json_handler)

    # Human-readable file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ConsoleFormatter())
        root_logger.addHandler(file_handler)
        updater_logger.addHandler(file_handler)

    return root_logger
