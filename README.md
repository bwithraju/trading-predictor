# Trading Predictor

A production-ready algorithmic trading prediction system with ML signals, technical analysis, and risk management.

## Features

- 📊 **Technical Analysis** — SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, ATR
- 🤖 **ML Predictions** — Random Forest classifier producing UP/DOWN/NO_SIGNAL
- ⚖️ **Risk Management** — Position sizing, stop-loss, take-profit, risk-reward ratio
- 🔄 **Backtesting** — Historical strategy simulation
- 🌐 **REST API** — FastAPI with auto-generated OpenAPI docs
- 🔌 **WebSocket** — Real-time signal streaming
- 🐳 **Docker** — Multi-stage build, Docker Compose, Kubernetes manifests
- 📈 **Monitoring** — Prometheus metrics endpoint

## Quick Start

```bash
git clone https://github.com/your-org/trading-predictor
cd trading-predictor
bash scripts/setup.sh
source venv/bin/activate
uvicorn main:app --reload
# Or with Docker:
docker compose up
```

The API is available at **http://localhost:8000**. Docs at **http://localhost:8000/docs**.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Liveness |
| GET | `/health` | Health + DB status |
| POST | `/predict` | ML prediction + risk |
| POST | `/risk` | Risk calculator |
| POST | `/backtest` | Historical backtest |
| GET | `/indicators/{symbol}` | Technical indicators |
| POST | `/train` | Train ML model |
| GET | `/metrics` | Prometheus metrics |
| WS | `/ws/updates` | Real-time WebSocket |

## Configuration

Copy `.env.example` to `.env` and edit. See [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Tests

```bash
pytest tests/ -v
```

## Documentation

- [Installation](docs/INSTALLATION.md) | [API Reference](docs/API.md) | [Trading Guide](docs/TRADING_GUIDE.md)
- [Architecture](docs/ARCHITECTURE.md) | [Deployment](docs/DEPLOYMENT.md) | [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

MIT
