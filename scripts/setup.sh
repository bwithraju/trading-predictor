#!/usr/bin/env bash
# Automated environment setup for trading-predictor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "==> Trading Predictor Setup"

# Python version check
python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" || {
  echo "ERROR: Python 3.10 or higher is required."
  exit 1
}

# Virtual environment
if [ ! -d "venv" ]; then
  echo "==> Creating virtual environment"
  python3 -m venv venv
fi

source venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create runtime directories
mkdir -p cache models

# Copy .env if missing
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — edit it with your credentials"
fi

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source venv/bin/activate"
echo "Then start the API with:"
echo "  uvicorn main:app --reload"
