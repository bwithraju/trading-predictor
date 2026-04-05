"""Unit tests for src/backtesting/."""
import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine


def _make_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestBacktestEngine:
    def test_run_returns_result(self):
        df = _make_df(300)
        engine = BacktestEngine(lookback=50, confidence_threshold=0.0, min_indicator_alignment=0)
        result = engine.run(df, symbol="BT_TEST")
        assert result.symbol == "BT_TEST"
        assert result.total_trades >= 0

    def test_win_rate_between_0_and_1(self):
        df = _make_df(300)
        engine = BacktestEngine(lookback=50, confidence_threshold=0.0, min_indicator_alignment=0)
        result = engine.run(df, symbol="BT_TEST2")
        assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self):
        df = _make_df(300)
        engine = BacktestEngine(lookback=50, confidence_threshold=0.0, min_indicator_alignment=0)
        result = engine.run(df, symbol="BT_TEST3")
        assert result.max_drawdown_pct >= 0.0

    def test_no_trades_result(self):
        """With very high threshold no trades should be taken."""
        df = _make_df(300)
        engine = BacktestEngine(
            lookback=50,
            confidence_threshold=0.999,  # impossible threshold
            min_indicator_alignment=100,  # impossible alignment
        )
        result = engine.run(df, symbol="NO_TRADES")
        assert result.total_trades == 0
        assert result.win_rate == 0.0
