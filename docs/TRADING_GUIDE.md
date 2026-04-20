# Trading Guide

## Signal Interpretation

| Signal | Meaning | Action |
|--------|---------|--------|
| `UP` | Model predicts price increase | Consider LONG entry |
| `DOWN` | Model predicts price decrease | Consider SHORT entry |
| `NO_SIGNAL` | Insufficient confidence | Stay flat / wait |

A signal is only emitted when model confidence ≥ `CONFIDENCE_THRESHOLD` (default 70%) **and** at least `MIN_INDICATOR_ALIGNMENT` (default 3) technical indicators agree.

## Risk Management

- **Risk Percent** — percentage of account risked per trade (default 2%).
- **Stop Loss** — placed at `entry_price ± SL_PERCENT` in the direction of the trade.
- **Take Profit** — placed at `stop_loss_distance × REWARD_RATIO` from entry.
- **Position Size** — calculated as `(account_size × risk_percent) / (entry_price - stop_loss)`.

## Technical Indicators Used

| Indicator | Role |
|---|---|
| SMA 20 / 50 | Trend direction |
| EMA 12 / 26 | Momentum |
| MACD | Trend momentum crossover |
| RSI (14) | Overbought / oversold |
| Bollinger Bands | Volatility regime |
| Stochastic | Short-term reversal |
| ATR (14) | Volatility / stop sizing |

## Backtesting Strategy

The built-in backtest engine:
1. Generates UP/DOWN signals over the historical period.
2. Enters at the next open after a signal.
3. Exits at stop-loss or take-profit (whichever is hit first).
4. Reports: total trades, win rate, total return, max drawdown, Sharpe ratio.

## Disclaimer

This software is for educational and research purposes only. It does **not** constitute financial advice. Always perform your own due diligence before trading.
