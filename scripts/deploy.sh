#!/usr/bin/env bash
# Deploy trading-predictor via Docker Compose.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TARGET="${1:-production}"
echo "==> Deploying target: $TARGET"

docker compose build --target "$TARGET"
docker compose up -d

echo ""
echo "API available at http://localhost:8000"
echo "Health check: http://localhost:8000/health"
