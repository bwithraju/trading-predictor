# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Docker & Docker Compose for containerised deployment

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/trading-predictor
cd trading-predictor

# 2. Run the automated setup script
bash scripts/setup.sh

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings

# 5. Start the API server
uvicorn main:app --reload
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Listen port |
| `DEBUG` | `false` | Enable reload/debug logging |
| `DB_URL` | `sqlite:///trading_predictor.db` | SQLAlchemy database URL |
| `BINANCE_API_KEY` | — | Binance API key for crypto data |
| `BINANCE_SECRET` | — | Binance API secret |
| `CONFIDENCE_THRESHOLD` | `0.70` | Minimum model confidence for a signal |
| `DEFAULT_RISK_PERCENT` | `2.0` | Default account risk per trade (%) |

## Docker Setup

```bash
# Build and start all services
docker compose up --build

# Production build
docker compose build --target production
docker compose up -d
```

## Verifying the Install

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","db":"ok",...}
```
