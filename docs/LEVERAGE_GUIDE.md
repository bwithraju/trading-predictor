# Leverage Prediction System – User Guide

## Overview

The **Leverage Prediction System** is an intelligent engine that dynamically recommends safe leverage ratios based on real-time market conditions.  Instead of using a fixed leverage multiplier, it analyses volatility, trend strength, drawdown patterns, and momentum to suggest the safest and most profitable leverage for the current environment.

---

## Quick Start

```python
import numpy as np
from src.leverage.model import LeverageModel

# Prepare your OHLCV data (oldest bar first, minimum 30 bars)
close = np.array([...])   # close prices
high  = np.array([...])   # high prices
low   = np.array([...])   # low prices

model = LeverageModel()

recommendation = model.recommend(
    close=close,
    high=high,
    low=low,
    symbol="BTC/USDT",
    timeframe="1h",
    account_size=1000.0,   # your equity in USD
)

print(f"Recommended leverage : {recommendation.recommended_leverage}×")
print(f"Volatility tier      : {recommendation.volatility_tier.value}")
print(f"Confidence           : {recommendation.confidence_score:.0f}%")
print(f"Liquidation risk     : {recommendation.liquidation_risk_pct:.1f}%")
print(f"Buying power         : ${recommendation.buying_power:,.2f}")
```

---

## Volatility Tiers

| Tier | Volatility Ratio | Max Leverage | Risk Level | Description |
|------|-----------------|--------------|------------|-------------|
| EXTREME | > 1.5× | 1:1 | Extreme Risk | Unpredictable moves |
| HIGH | 1.0–1.5× | 2:1 | High Risk | Frequent large moves |
| NORMAL | 0.7–1.0× | 4:1 | Moderate Risk | Predictable patterns |
| LOW | 0.5–0.7× | 8:1 | Low Risk | Stable trending market |
| EXTREME_LOW | < 0.5× | 15:1 | Lowest Risk | Very stable conditions |

The **volatility ratio** is the current 20-period annualised volatility divided by the 100-period baseline.

---

## API Endpoints

Mount the router in your FastAPI application:

```python
from fastapi import FastAPI
from src.api.leverage_endpoints import router as leverage_router

app = FastAPI()
app.include_router(leverage_router)
```

### `POST /leverage/recommend`

Generate a leverage recommendation.

**Request body:**

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "account_size": 1000.0,
  "close_prices": [50000, 50500, ...],
  "high_prices":  [50200, 50700, ...],
  "low_prices":   [49800, 50300, ...],
  "risk_percent": 2.0,
  "stop_loss_percent": 5.0
}
```

**Response:**

```json
{
  "recommended_leverage": 4,
  "max_safe_leverage": 4,
  "aggressive_leverage": 8,
  "confidence_score": 72.5,
  "safety_score": 85.0,
  "liquidation_risk_pct": 15.0,
  "volatility_tier": "NORMAL",
  "buying_power": 4000.0,
  "max_position_size": 400.0,
  "daily_stop_loss": 50.0,
  "risk_level": "MODERATE"
}
```

### `GET /leverage/status`

Returns model load status and supported leverage range.

### `POST /leverage/adjust`

Approve or reject a requested leverage change.

### `GET /leverage/analysis/{symbol}?close_prices=...`

Volatility tier and feature analysis for a symbol.

### `POST /leverage/backtest`

Run a simplified leverage backtest over historical price data.

### `GET /leverage/insights`

Market overview and tier threshold reference.

---

## Training a Custom Model

```python
from src.leverage.trainer import LeverageTrainer

trainer = LeverageTrainer()

# Option 1: Use synthetic data (for smoke tests)
model = trainer.train_from_synthetic(n_bars=2000)

# Option 2: Use real historical data
close, high, low = load_your_data()
X, y_cls, y_reg = trainer.build_training_data([(close, high, low)])
model = trainer.train(X, y_cls, y_reg)

# Save the trained model
model.save_model()
```

---

## Risk Adjustment Factors

The system automatically reduces leverage when:

| Condition | Action |
|-----------|--------|
| Drawdown > 20% | Reduce leverage by 2 |
| Drawdown > 10% | Reduce leverage by 1 |
| ADX < 20 (weak trend) | Reduce leverage by 1 |
| Volatility spike detected | Warning issued |
| Safety checks fail | Reduce to 1× |

---

## Safety Checks

Before issuing any recommendation, the following checks run:

1. ✅ Account size ≥ minimum ($10 by default)
2. ✅ Sufficient price data (≥30 bars)
3. ✅ Volatility data available (no NaN)
4. ✅ No active volatility spike
5. ✅ Drawdown within threshold (< 25%)
6. ✅ Leverage within hard cap (≤ 20×)
7. ✅ Liquidation risk < 40%

If any **CRITICAL** check fails, the recommendation is reduced to 1× (no leverage).

---

## Small Account Guidance ($100 – $1,000)

For small accounts, the position-sizing math often limits leverage even in low-volatility conditions.  Use:

- **Risk per trade**: 1–2%
- **Stop-loss distance**: 3–5%
- **Max positions**: 1–3

At $100 with 2% risk and 5% stop-loss, the safe position size is $40 – giving an implied max leverage of 0.4×, which the system rounds up to 1×.

To unlock higher leverage on small accounts:
- Widen your stop-loss (increases the allowed position size)
- Increase the risk-per-trade setting (not recommended for beginners)

---

## Running Tests

```bash
cd /path/to/trading-predictor
python -m pytest tests/ -v --tb=short
```
