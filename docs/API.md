# API Reference

Base URL: `http://localhost:8000`

All request/response bodies are JSON unless noted.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + DB status |
| POST | `/predict/stock` | ML prediction for stocks |
| POST | `/predict/crypto` | ML prediction for crypto |
| POST | `/calculate-risk` | Risk calculator |
| POST | `/backtest` | Historical backtest |
| GET | `/indicators/{symbol}` | Technical indicators |
| POST | `/train/{symbol}` | Train ML model |
| GET | `/metrics` | Prometheus metrics |
| WS | `/ws/updates` | Real-time WebSocket |

---

## Health

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy", "db": "ok", "websocket_connections": 0}
```

---

## Prediction

### POST /predict/stock

```bash
curl -X POST http://localhost:8000/predict/stock \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","entry_price":180.0,"risk_percent":2.0,"account_size":10000}'
```

Response:
```json
{
  "symbol": "AAPL",
  "prediction": "UP",
  "confidence": 0.83,
  "direction": "LONG",
  "stop_loss": 176.40,
  "take_profit": 187.20,
  "position_size": 139,
  "indicators": {...}
}
```

### POST /predict/crypto

```bash
curl -X POST http://localhost:8000/predict/crypto \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","entry_price":42000.0}'
```

---

## Risk Calculator

### POST /calculate-risk

```bash
curl -X POST http://localhost:8000/calculate-risk \
  -H "Content-Type: application/json" \
  -d '{"entry_price":180.0,"direction":"LONG","risk_percent":2.0,"account_size":10000}'
```

---

## Backtesting

### POST /backtest

```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","period":"2y","risk_percent":1.5}'
```

---

## Technical Indicators

### GET /indicators/{symbol}

```bash
curl "http://localhost:8000/indicators/AAPL?interval=1d&period=6mo"
```

---

## Model Training

### POST /train/{symbol}

```bash
curl -X POST "http://localhost:8000/train/AAPL?asset_type=stock"
```

---

## Monitoring

### GET /metrics

Prometheus text format metrics:

```bash
curl http://localhost:8000/metrics
```

---

## WebSocket

### WS /ws/updates

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/updates");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({action: "subscribe", symbol: "AAPL"}));
```
