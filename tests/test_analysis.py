"""Tests for technical analysis and signal generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import TechnicalAnalysis
from src.analysis.signals import SignalGenerator


class TestTechnicalAnalysis:
    ta = TechnicalAnalysis()

    def test_sma(self, ohlcv_df):
        result = self.ta.sma(ohlcv_df["close"], 20)
        assert len(result) == len(ohlcv_df)
        # First 19 rows should be NaN
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().all()

    def test_ema(self, ohlcv_df):
        result = self.ta.ema(ohlcv_df["close"], 12)
        assert len(result) == len(ohlcv_df)
        assert not result.isna().any()

    def test_rsi(self, ohlcv_df):
        result = self.ta.rsi(ohlcv_df["close"])
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd(self, ohlcv_df):
        macd, signal, hist = self.ta.macd(ohlcv_df["close"])
        assert len(macd) == len(ohlcv_df)
        assert len(signal) == len(ohlcv_df)
        assert len(hist) == len(ohlcv_df)

    def test_bollinger_bands(self, ohlcv_df):
        upper, mid, lower = self.ta.bollinger_bands(ohlcv_df["close"])
        valid_upper = upper.dropna()
        valid_lower = lower.dropna()
        assert (valid_upper >= valid_lower).all()

    def test_atr(self, ohlcv_df):
        result = self.ta.atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_add_all_indicators(self, ohlcv_df):
        df = self.ta.add_all_indicators(ohlcv_df.copy())
        expected_cols = ["sma_20", "sma_50", "rsi", "macd", "bb_upper", "atr"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_add_all_indicators_short(self, short_ohlcv_df):
        """Short DataFrames should return unchanged (warning logged)."""
        df = self.ta.add_all_indicators(short_ohlcv_df.copy())
        assert df.equals(short_ohlcv_df)

    def test_get_current_indicators(self, ohlcv_df):
        indicators = self.ta.get_current_indicators(ohlcv_df)
        assert isinstance(indicators, dict)
        assert "close" in indicators or "rsi" in indicators


class TestSignalGenerator:
    sg = SignalGenerator()

    def test_rsi_signal_oversold(self):
        assert self.sg._rsi_signal(25) == 1

    def test_rsi_signal_overbought(self):
        assert self.sg._rsi_signal(75) == -1

    def test_rsi_signal_neutral(self):
        assert self.sg._rsi_signal(50) == 0

    def test_macd_signal_bullish(self):
        assert self.sg._macd_signal(0.5, 0.3) == 1

    def test_macd_signal_bearish(self):
        assert self.sg._macd_signal(0.2, 0.5) == -1

    def test_generate_signals(self, ohlcv_df):
        from src.analysis.technical import TechnicalAnalysis
        ta = TechnicalAnalysis()
        df = ta.add_all_indicators(ohlcv_df.copy())
        last = df.iloc[-1].to_dict()
        sigs = self.sg.generate_signals(last)
        assert "bull_count" in sigs
        assert "bear_count" in sigs
        assert "direction" in sigs
        assert sigs["direction"] in ("UP", "DOWN", "NEUTRAL")

    def test_generate_from_df(self, ohlcv_df):
        sigs = self.sg.generate_from_df(ohlcv_df)
        assert isinstance(sigs, dict)
        assert "direction" in sigs
