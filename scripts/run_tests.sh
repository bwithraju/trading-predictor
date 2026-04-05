#!/usr/bin/env bash
# Run the test suite with optional coverage reporting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

COVERAGE="${COVERAGE:-false}"

if [ "$COVERAGE" = "true" ]; then
  echo "==> Running tests with coverage"
  pip install pytest-cov -q
  pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
else
  echo "==> Running tests"
  pytest tests/ -v --tb=short
fi
