"""
Backup utilities for velocity trading system.

Provides automatic backup functionality to prevent data loss:
- Database backups before destructive operations
- Config file backups before changes
- Strategy bundle backups before migrations

All backups are timestamped and stored in a dedicated backup directory.
"""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import json

# Base directories
VELOCITY_TRADING_DIR = Path(__file__).parent.parent
PROJECT_ROOT = VELOCITY_TRADING_DIR.parent
BACKUP_ROOT = PROJECT_ROOT / "backups"


def get_backup_dir(category: str = "general") -> Path:
    """
    Get backup directory for a specific category.

    Args:
        category: Backup category (e.g., 'database', 'config', 'migration')

    Returns:
        Path to backup directory (created if doesn't exist)
    """
    backup_dir = BACKUP_ROOT / category
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def generate_backup_name(original_name: str, suffix: str = "") -> str:
    """
    Generate timestamped backup filename.

    Args:
        original_name: Original filename
        suffix: Optional suffix to add (e.g., 'pre_migration')

    Returns:
        Backup filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(original_name)
    if suffix:
        return f"{base}_{suffix}_{timestamp}{ext}"
    return f"{base}_{timestamp}{ext}"


def backup_file(
    source_path: str,
    category: str = "general",
    suffix: str = "",
    preserve_original: bool = True
) -> Optional[str]:
    """
    Create a backup of a file.

    Args:
        source_path: Path to file to backup
        category: Backup category for organization
        suffix: Optional suffix for backup name
        preserve_original: If True, copy. If False, move.

    Returns:
        Path to backup file, or None if source doesn't exist
    """
    source = Path(source_path)
    if not source.exists():
        return None

    backup_dir = get_backup_dir(category)
    backup_name = generate_backup_name(source.name, suffix)
    backup_path = backup_dir / backup_name

    if preserve_original:
        shutil.copy2(source, backup_path)
    else:
        shutil.move(source, backup_path)

    print(f"   [Backup] {source.name} -> {backup_path}")
    return str(backup_path)


def backup_database(
    db_path: str,
    reason: str = "auto",
    strategy_name: str = None
) -> Optional[str]:
    """
    Create a backup of a SQLite database.

    Uses SQLite's backup API for safe, consistent backups even if
    the database is in use.

    Args:
        db_path: Path to database file
        reason: Reason for backup (used in filename)
        strategy_name: Optional strategy name for organization

    Returns:
        Path to backup file, or None if source doesn't exist
    """
    source = Path(db_path)
    if not source.exists():
        return None

    # Organize by strategy if provided
    category = f"database/{strategy_name}" if strategy_name else "database"
    backup_dir = get_backup_dir(category)
    backup_name = generate_backup_name(source.name, reason)
    backup_path = backup_dir / backup_name

    # Use SQLite backup API for safe backup
    try:
        source_conn = sqlite3.connect(db_path)
        backup_conn = sqlite3.connect(str(backup_path))
        source_conn.backup(backup_conn)
        source_conn.close()
        backup_conn.close()
        print(f"   [DB Backup] {source.name} -> {backup_path}")
        return str(backup_path)
    except Exception as e:
        print(f"   [DB Backup Error] {e}")
        # Fallback to file copy
        return backup_file(db_path, category, reason)


def backup_strategy_bundle(
    bundle_path: str,
    reason: str = "auto"
) -> Optional[str]:
    """
    Create a backup of an entire strategy bundle directory.

    Args:
        bundle_path: Path to strategy bundle directory
        reason: Reason for backup

    Returns:
        Path to backup directory, or None if source doesn't exist
    """
    source = Path(bundle_path)
    if not source.exists() or not source.is_dir():
        return None

    backup_dir = get_backup_dir("bundles")
    backup_name = generate_backup_name(source.name, reason)
    backup_path = backup_dir / backup_name

    shutil.copytree(source, backup_path)
    print(f"   [Bundle Backup] {source.name} -> {backup_path}")
    return str(backup_path)


def backup_config(
    config_path: str,
    reason: str = "auto"
) -> Optional[str]:
    """
    Create a backup of a config file.

    Args:
        config_path: Path to config file
        reason: Reason for backup

    Returns:
        Path to backup file, or None if source doesn't exist
    """
    return backup_file(config_path, "config", reason)


def backup_all_strategy_files(
    strategy_name: str,
    reason: str = "auto"
) -> Dict[str, str]:
    """
    Backup all files associated with a strategy.

    This includes:
    - Database
    - Config (in bundle)
    - Locked backtest
    - Trade history
    - Trade state
    - Daily alerts

    Args:
        strategy_name: Name of the strategy
        reason: Reason for backup

    Returns:
        Dictionary of {file_type: backup_path}
    """
    backups = {}

    # Database
    db_path = PROJECT_ROOT / "velocity_trading" / "db" / f"{strategy_name}.db"
    if db_path.exists():
        backups['database'] = backup_database(str(db_path), reason, strategy_name)

    # Strategy bundle
    bundle_path = PROJECT_ROOT / "velocity_strategies" / strategy_name
    if bundle_path.exists():
        backups['bundle'] = backup_strategy_bundle(str(bundle_path), reason)

    # JSON files in project root
    json_files = [
        f"velocity_locked_backtest_{strategy_name}.json",
        f"velocity_trade_history_{strategy_name}.json",
        f"velocity_trade_state_{strategy_name}.json",
        f"daily_alerts_{strategy_name}.json"
    ]

    for json_file in json_files:
        json_path = PROJECT_ROOT / json_file
        if json_path.exists():
            file_type = json_file.replace(f"_{strategy_name}.json", "").replace("velocity_", "")
            backups[file_type] = backup_file(str(json_path), f"json/{strategy_name}", reason)

    return backups


def list_backups(
    category: str = None,
    strategy_name: str = None,
    limit: int = 20
) -> List[Dict]:
    """
    List available backups.

    Args:
        category: Filter by category
        strategy_name: Filter by strategy name
        limit: Maximum number of results

    Returns:
        List of backup info dictionaries
    """
    backups = []

    if category:
        search_dirs = [get_backup_dir(category)]
    else:
        search_dirs = [BACKUP_ROOT]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = Path(root) / file

                # Filter by strategy name if provided
                if strategy_name and strategy_name not in file:
                    continue

                backups.append({
                    'path': str(file_path),
                    'name': file,
                    'category': Path(root).relative_to(BACKUP_ROOT).parts[0] if BACKUP_ROOT in file_path.parents else 'root',
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })

    # Sort by modified time, newest first
    backups.sort(key=lambda x: x['modified'], reverse=True)
    return backups[:limit]


def restore_from_backup(
    backup_path: str,
    destination_path: str,
    create_backup_of_current: bool = True
) -> bool:
    """
    Restore a file from backup.

    Args:
        backup_path: Path to backup file
        destination_path: Where to restore to
        create_backup_of_current: If True, backup current file first

    Returns:
        True if successful
    """
    backup = Path(backup_path)
    dest = Path(destination_path)

    if not backup.exists():
        print(f"   [Restore Error] Backup not found: {backup_path}")
        return False

    # Backup current file if it exists
    if create_backup_of_current and dest.exists():
        backup_file(str(dest), "pre_restore", "pre_restore")

    # Restore
    if backup.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(backup, dest)
    else:
        shutil.copy2(backup, dest)

    print(f"   [Restored] {backup.name} -> {dest}")
    return True


# Convenience function for pre-migration backup
def pre_migration_backup(strategy_name: str) -> Dict[str, str]:
    """
    Create a full backup before any migration/rename operation.

    This should be called BEFORE any file renaming or config changes.

    Args:
        strategy_name: Current strategy name (before rename)

    Returns:
        Dictionary of backup paths
    """
    print(f"\n{'='*60}")
    print(f"Creating pre-migration backup for: {strategy_name}")
    print(f"{'='*60}")

    backups = backup_all_strategy_files(strategy_name, "pre_migration")

    print(f"\nBackup complete. {len(backups)} files backed up.")
    print(f"Backups stored in: {BACKUP_ROOT}")
    print(f"{'='*60}\n")

    return backups
