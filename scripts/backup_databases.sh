#!/bin/bash
#
# Backup all trading databases with timestamped label.
#
# Usage:
#   ./scripts/backup_databases.sh "pre_phase_c"
#   ./scripts/backup_databases.sh "before_migration"
#
# Creates: backups/<label>_YYYYMMDD_HHMMSS/
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
NC='\033[0m' # No Color

# Get label from argument
LABEL="${1:-manual}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/${LABEL}_${TIMESTAMP}"

echo "=========================================="
echo "DATABASE BACKUP"
echo "=========================================="
echo ""
echo "Label:      $LABEL"
echo "Timestamp:  $TIMESTAMP"
echo "Source:     $DB_DIR"
echo "Dest:       $BACKUP_DIR"
echo ""

# Check if source directory exists
if [ ! -d "$DB_DIR" ]; then
    echo -e "${RED}Error: Database directory not found: $DB_DIR${NC}"
    exit 1
fi

# Count databases
DB_COUNT=$(find "$DB_DIR" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')

if [ "$DB_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No .db files found in $DB_DIR${NC}"
    exit 0
fi

echo "Found $DB_COUNT database(s) to backup"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Copy all database files
echo "Copying databases..."
COPIED=0
TOTAL_SIZE=0

for db_file in "$DB_DIR"/*.db; do
    if [ -f "$db_file" ]; then
        filename=$(basename "$db_file")
        cp "$db_file" "$BACKUP_DIR/$filename"
        size=$(ls -lh "$db_file" | awk '{print $5}')
        echo "  - $filename ($size)"
        ((COPIED++))
    fi
done

# Calculate total backup size
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | awk '{print $1}')

echo ""
echo "=========================================="
echo -e "${GREEN}BACKUP COMPLETE${NC}"
echo "=========================================="
echo "Files backed up: $COPIED"
echo "Total size:      $TOTAL_SIZE"
echo "Location:        $BACKUP_DIR"
echo ""

# Write metadata
cat > "$BACKUP_DIR/backup_info.txt" << EOF
Backup Label: $LABEL
Timestamp: $TIMESTAMP
Created: $(date)
Files: $COPIED
Source: $DB_DIR
EOF

echo "Metadata written to: $BACKUP_DIR/backup_info.txt"
echo ""

# List recent backups
echo "Recent backups:"
ls -dt "$BACKUP_ROOT"/*/ 2>/dev/null | head -5 | while read dir; do
    if [ -f "$dir/backup_info.txt" ]; then
        label=$(basename "$dir")
        size=$(du -sh "$dir" | awk '{print $1}')
        echo "  - $label ($size)"
    fi
done

echo ""
echo "To restore: ./scripts/restore_databases.sh \"${LABEL}_${TIMESTAMP}\""
