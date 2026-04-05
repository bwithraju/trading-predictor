#!/usr/bin/env bash
# Back up the SQLite database.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DB_FILE="${DB_FILE:-trading_predictor.db}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$BACKUP_DIR"

if [ ! -f "$DB_FILE" ]; then
  echo "No database found at $DB_FILE — nothing to back up."
  exit 0
fi

DEST="$BACKUP_DIR/${DB_FILE%.db}_${TIMESTAMP}.db"
cp "$DB_FILE" "$DEST"
echo "Backup saved to $DEST"

# Keep only the last 30 backups
find "$BACKUP_DIR" -name "*.db" -type f | sort | head -n -30 | xargs -r rm --
