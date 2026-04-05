# Leverage Recommendations Reference

## Volatility Tier Reference

### Tier 1 – EXTREME VOLATILITY (Extreme Risk)
- **Condition**: Volatility ratio > 1.5× historical average
- **Recommended Leverage**: 1:1 (no leverage)
- **Liquidation Risk**: 40%+
- **Rationale**: Price moves are too unpredictable; any leverage dramatically raises liquidation probability.

### Tier 2 – HIGH VOLATILITY (High Risk)
- **Condition**: Volatility ratio 1.0–1.5×
- **Recommended Leverage**: 2:1 maximum
- **Liquidation Risk**: 20–30%
- **Rationale**: Frequent large moves can quickly breach stop-losses.

### Tier 3 – NORMAL VOLATILITY (Moderate Risk)
- **Condition**: Volatility ratio 0.7–1.0×
- **Recommended Leverage**: 4:1 maximum
- **Liquidation Risk**: 10–15%
- **Rationale**: Market behaves predictably; moderate leverage captures upside without excessive risk.

### Tier 4 – LOW VOLATILITY (Low Risk)
- **Condition**: Volatility ratio 0.5–0.7×
- **Recommended Leverage**: 8:1 maximum
- **Liquidation Risk**: 5–10%
- **Rationale**: Stable trending conditions; higher leverage acceptable with tight risk controls.

### Tier 5 – EXTREME LOW VOLATILITY (Lowest Risk)
- **Condition**: Volatility ratio < 0.5×
- **Recommended Leverage**: 15:1 maximum
- **Liquidation Risk**: 2–5%
- **Rationale**: Very stable market; leverage is safe but be aware of sudden regime changes.

---

## Dynamic Adjustment Rules

```
Win rate trending down?      → Reduce leverage by 1–2
Drawdown > 20%?              → Reduce leverage by 2
Drawdown > 10%?              → Reduce leverage by 1
Volatility spike detected?   → Warning; consider reducing
Trend ADX < 20?              → Reduce leverage by 1
Safety check fails?          → Override to 1× (no leverage)
```

---

## Account-Size Recommendations

| Account Size | Risk% | Stop-Loss% | Suggested Max Leverage |
|-------------|-------|-----------|----------------------|
| $100        | 1%    | 3%        | 1× (position-sizing limits) |
| $500        | 1%    | 3%        | 1–2× |
| $1,000      | 2%    | 5%        | 1–4× (tier-dependent) |
| $5,000      | 2%    | 5%        | Up to tier maximum |
| $10,000+    | 2%    | 5%        | Full tier maximum |

> **Note**: Position-sizing math limits effective leverage for small accounts.
> A $100 account at 2% risk / 5% stop = $40 position = 0.4× implied leverage → rounded to 1×.

---

## Liquidation Risk Formula

```
max_adverse_move  = volatility × (2 - trend_strength)
margin            = 1 / leverage
liquidation_risk  = (max_adverse_move / margin) × 100%

Risk Levels:
  > 30%  → CRITICAL
  > 15%  → HIGH
  > 5%   → MODERATE
  ≤ 5%   → LOW
```

---

## Confidence Score Interpretation

| Score | Meaning |
|-------|---------|
| 80–100 | High confidence; signals strongly aligned |
| 60–79 | Moderate confidence; proceed with caution |
| 40–59 | Low confidence; reduce position size |
| < 40  | Very low confidence; avoid leverage |

---

## API Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/leverage/recommend` | POST | Full leverage recommendation |
| `/leverage/status` | GET | Model status |
| `/leverage/adjust` | POST | Approve/reject leverage change |
| `/leverage/analysis/{symbol}` | GET | Volatility tier analysis |
| `/leverage/backtest` | POST | Historical leverage simulation |
| `/leverage/insights` | GET | Market overview and tier info |
