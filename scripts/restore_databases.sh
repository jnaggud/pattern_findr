#!/bin/bash
#
# Restore trading databases from a backup.
#
# Usage:
#   ./scripts/restore_databases.sh "pre_phase_c_20260205_143000"
#   ./scripts/restore_databases.sh --list  # List available backups
#
# WARNING: This will overwrite current databases!
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_DIR="$PROJECT_ROOT/velocity_trading/db"
BACKUP_ROOT="$PROJECT_ROOT/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Show available backups
list_backups() {
    echo "Available backups:"
    echo ""

    if [ ! -d "$BACKUP_ROOT" ]; then
        echo "  No backups found (backup directory doesn't exist)"
        return
    fi

    for backup_dir in $(ls -dt "$BACKUP_ROOT"/*/ 2>/dev/null); do
        if [ -f "$backup_dir/backup_info.txt" ]; then
            label=$(basename "$backup_dir")
            size=$(du -sh "$backup_dir" | awk '{print $1}')
            files=$(find "$backup_dir" -name "*.db" | wc -l | tr -d ' ')
            created=$(grep "Created:" "$backup_dir/backup_info.txt" | cut -d: -f2- | xargs)
            echo -e "  ${CYAN}$label${NC}"
            echo "    Size: $size, Files: $files"
            echo "    Created: $created"
            echo ""
        fi
    done
}

# Handle --list flag
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    list_backups
    exit 0
fi

# Get backup label from argument
BACKUP_LABEL="$1"

if [ -z "$BACKUP_LABEL" ]; then
    echo -e "${RED}Error: No backup label provided${NC}"
    echo ""
    echo "Usage: $0 <backup_label>"
    echo "       $0 --list"
    echo ""
    list_backups
    exit 1
fi

# Find backup directory
BACKUP_DIR="$BACKUP_ROOT/$BACKUP_LABEL"

if [ ! -d "$BACKUP_DIR" ]; then
    echo -e "${RED}Error: Backup not found: $BACKUP_DIR${NC}"
    echo ""
    list_backups
    exit 1
fi

# Count files
FILE_COUNT=$(find "$BACKUP_DIR" -name "*.db" | wc -l | tr -d ' ')

if [ "$FILE_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No .db files in backup${NC}"
    exit 1
fi

echo "=========================================="
echo "DATABASE RESTORE"
echo "=========================================="
echo ""
echo "Backup:     $BACKUP_LABEL"
echo "Source:     $BACKUP_DIR"
echo "Dest:       $DB_DIR"
echo "Files:      $FILE_COUNT"
echo ""

# Show backup info
if [ -f "$BACKUP_DIR/backup_info.txt" ]; then
    echo "Backup info:"
    cat "$BACKUP_DIR/backup_info.txt" | sed 's/^/  /'
    echo ""
fi

# Confirm before proceeding
echo -e "${YELLOW}WARNING: This will overwrite current databases!${NC}"
echo ""
read -p "Are you sure you want to restore? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Restore cancelled."
    exit 0
fi

echo ""
echo "Restoring databases..."

# Ensure destination exists
mkdir -p "$DB_DIR"

# Copy files
RESTORED=0
for db_file in "$BACKUP_DIR"/*.db; do
    if [ -f "$db_file" ]; then
        filename=$(basename "$db_file")
        cp "$db_file" "$DB_DIR/$filename"
        echo "  - $filename"
        ((RESTORED++))
    fi
done

echo ""
echo "=========================================="
echo -e "${GREEN}RESTORE COMPLETE${NC}"
echo "=========================================="
echo "Files restored: $RESTORED"
echo ""
echo "Please restart any running traders to use the restored data."
