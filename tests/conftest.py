"""Shared pytest fixtures for the trading-predictor test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Synthetic market data helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    volume = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """200-row synthetic OHLCV DataFrame."""
    return _make_ohlcv()


@pytest.fixture
def short_ohlcv_df() -> pd.DataFrame:
    """20-row synthetic OHLCV DataFrame (below most indicator windows)."""
    return _make_ohlcv(n=20)


@pytest.fixture
def long_ohlcv_df() -> pd.DataFrame:
    """500-row synthetic OHLCV DataFrame for backtesting."""
    return _make_ohlcv(n=500, seed=99)


# ---------------------------------------------------------------------------
# Mock external API calls so tests never hit the network
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_yfinance(monkeypatch):
    """Patch yfinance.download to return synthetic data."""
    mock_data = _make_ohlcv()
    mock_data.columns = [c.capitalize() for c in mock_data.columns]

    with patch("yfinance.download", return_value=mock_data):
        yield


@pytest.fixture(autouse=True)
def mock_ccxt(monkeypatch):
    """Patch ccxt exchange fetch_ohlcv calls."""
    synthetic = _make_ohlcv()
    ohlcv_rows = [
        [int(ts.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
        for ts, row in synthetic.iterrows()
    ]
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = ohlcv_rows

    with patch("ccxt.binance", return_value=mock_exchange):
        yield
