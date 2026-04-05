# Architecture Overview

## High-Level Diagram

```
┌─────────────────────────────────────────────────┐
│                   Clients                        │
│  (REST / WebSocket / Frontend Dashboard)         │
└───────────────────┬─────────────────────────────┘
                    │ HTTP / WS
┌───────────────────▼─────────────────────────────┐
│              nginx (reverse proxy)               │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│         FastAPI Application (main.py)            │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │  REST API   │  │  WebSocket Manager        │  │
│  │ (endpoints) │  │  (real-time updates)      │  │
│  └──────┬──────┘  └──────────────────────────┘  │
│         │                                        │
│  ┌──────▼───────────────────────────────────┐   │
│  │              Core Modules                │   │
│  │  data/  analysis/  models/  risk/        │   │
│  │  backtesting/  auth/  db/  utils/        │   │
│  └──────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  ┌──────────┐          ┌──────────────┐
  │ SQLite / │          │  Redis Cache │
  │ Postgres │          │  (optional)  │
  └──────────┘          └──────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/data/` | Fetch OHLCV data (yfinance, ccxt), cache, persist |
| `src/analysis/` | Technical indicators and signal generation |
| `src/models/` | Feature engineering, model training & prediction |
| `src/risk/` | Stop-loss, take-profit, position sizing |
| `src/backtesting/` | Historical strategy simulation |
| `src/api/` | FastAPI route handlers |
| `src/auth/` | JWT / API-key auth, rate limiting |
| `src/websocket/` | WebSocket connection management |
| `src/db/` | SQLAlchemy models and DB initialisation |
| `src/utils/` | Logging, input validators |

## Data Flow (Prediction Request)

1. Client POST `/predict` → `endpoints.py`
2. `DataFetcher` pulls OHLCV (cache → DB → external API)
3. `TechnicalAnalysis` computes indicators
4. `SignalGenerator` checks indicator alignment
5. `TradingPredictor` runs ML model → direction + confidence
6. `RiskCalculator` computes SL, TP, position size
7. Response returned to client
