import re
from typing import Optional


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a trading symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.upper().strip()
    if not re.match(r"^[A-Z0-9/\-\.]{1,20}$", symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol


def validate_price(price: float, name: str = "price") -> float:
    """Validate that a price is a positive number."""
    if price is None or price <= 0:
        raise ValueError(f"{name} must be a positive number, got {price}")
    return float(price)


def validate_risk_percent(risk_percent: float) -> float:
    """Validate risk percentage is between 0.01 and 100."""
    if risk_percent is None or not (0.01 <= risk_percent <= 100):
        raise ValueError(
            f"risk_percent must be between 0.01 and 100, got {risk_percent}"
        )
    return float(risk_percent)


def validate_ohlcv(data) -> bool:
    """Return True if DataFrame has the required OHLCV columns and non-empty."""
    required = {"open", "high", "low", "close", "volume"}
    if data is None or data.empty:
        return False
    cols = {c.lower() for c in data.columns}
    return required.issubset(cols)
