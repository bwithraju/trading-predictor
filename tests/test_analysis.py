"""Unit tests for src/analysis/."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import TechnicalAnalysis
from src.analysis.signals import SignalGenerator


def _make_df(n: int = 100) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestTechnicalAnalysis:
    ta = TechnicalAnalysis()

    def test_sma_length(self):
        df = _make_df(100)
        sma = self.ta.sma(df["close"], 20)
        assert len(sma) == 100
        assert sma.isna().sum() == 19  # first 19 rows are NaN

    def test_ema_length(self):
        df = _make_df(50)
        ema = self.ta.ema(df["close"], 12)
        assert len(ema) == 50
        assert not ema.isna().all()

    def test_macd_returns_three_series(self):
        df = _make_df(60)
        macd, signal, hist = self.ta.macd(df["close"])
        assert len(macd) == len(signal) == len(hist) == 60

    def test_rsi_range(self):
        df = _make_df(100)
        rsi = self.ta.rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_stochastic_range(self):
        df = _make_df(100)
        k, d = self.ta.stochastic(df["high"], df["low"], df["close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_bollinger_bands_structure(self):
        df = _make_df(50)
        upper, mid, lower = self.ta.bollinger_bands(df["close"])
        valid = ~(upper.isna() | mid.isna() | lower.isna())
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_atr_positive(self):
        df = _make_df(50)
        atr = self.ta.atr(df["high"], df["low"], df["close"])
        assert (atr.dropna() > 0).all()

    def test_add_all_indicators_columns(self):
        df = _make_df(100)
        enriched = self.ta.add_all_indicators(df.copy())
        for col in ["sma_20", "ema_12", "macd", "rsi", "stoch_k", "bb_upper", "atr"]:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_add_all_indicators_short_df_returns_unchanged(self):
        df = _make_df(20)
        result = self.ta.add_all_indicators(df.copy())
        # Should return without modification when too short
        assert "sma_20" not in result.columns

    def test_get_current_indicators_returns_dict(self):
        df = _make_df(100)
        ind = self.ta.get_current_indicators(df)
        assert isinstance(ind, dict)
        assert "close" in ind


class TestSignalGenerator:
    sg = SignalGenerator()

    def test_rsi_signal_oversold(self):
        assert self.sg._rsi_signal(25) == 1

    def test_rsi_signal_overbought(self):
        assert self.sg._rsi_signal(75) == -1

    def test_rsi_signal_neutral(self):
        assert self.sg._rsi_signal(50) == 0

    def test_macd_signal_bullish(self):
        assert self.sg._macd_signal(1.0, 0.5) == 1

    def test_macd_signal_bearish(self):
        assert self.sg._macd_signal(0.5, 1.0) == -1

    def test_generate_signals_returns_direction(self):
        df = _make_df(100)
        sigs = self.sg.generate_from_df(df)
        assert "direction" in sigs
        assert sigs["direction"] in ("UP", "DOWN", "NEUTRAL")

    def test_generate_signals_counts(self):
        df = _make_df(100)
        sigs = self.sg.generate_from_df(df)
        assert sigs["bull_count"] + sigs["bear_count"] <= 5
