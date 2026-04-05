"""Generate BUY/SELL/HOLD signals from technical indicators."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.analysis.technical import TechnicalAnalysis
from src.utils.logger import get_logger

logger = get_logger(__name__)

TA = TechnicalAnalysis()


class SignalGenerator:
    """Derive directional signals from indicator values."""

    # ------------------------------------------------------------------
    # Individual signal helpers  (return +1 bullish, -1 bearish, 0 neutral)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi_signal(rsi: Optional[float]) -> int:
        if rsi is None:
            return 0
        if rsi < 30:
            return 1   # oversold → bullish
        if rsi > 70:
            return -1  # overbought → bearish
        return 0

    @staticmethod
    def _macd_signal(macd: Optional[float], macd_signal: Optional[float]) -> int:
        if macd is None or macd_signal is None:
            return 0
        return 1 if macd > macd_signal else -1

    @staticmethod
    def _bb_signal(close: Optional[float], bb_upper: Optional[float], bb_lower: Optional[float]) -> int:
        if None in (close, bb_upper, bb_lower):
            return 0
        if close <= bb_lower:
            return 1
        if close >= bb_upper:
            return -1
        return 0

    @staticmethod
    def _sma_signal(close: Optional[float], sma_20: Optional[float], sma_50: Optional[float]) -> int:
        if None in (close, sma_20, sma_50):
            return 0
        if close > sma_20 > sma_50:
            return 1
        if close < sma_20 < sma_50:
            return -1
        return 0

    @staticmethod
    def _stoch_signal(stoch_k: Optional[float], stoch_d: Optional[float]) -> int:
        if None in (stoch_k, stoch_d):
            return 0
        if stoch_k < 20 and stoch_d < 20:
            return 1
        if stoch_k > 80 and stoch_d > 80:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    def generate_signals(self, indicators: dict) -> dict:
        """Evaluate all indicator signals and return a summary dict.

        Returns
        -------
        dict with keys:
            ``rsi_signal``, ``macd_signal``, ``bb_signal``,
            ``sma_signal``, ``stoch_signal``,
            ``bull_count``, ``bear_count``,
            ``composite``  (+1 / -1 / 0),
            ``direction``  ('UP' / 'DOWN' / 'NEUTRAL')
        """
        close = indicators.get("close")

        sigs = {
            "rsi_signal": self._rsi_signal(indicators.get("rsi")),
            "macd_signal": self._macd_signal(indicators.get("macd"), indicators.get("macd_signal")),
            "bb_signal": self._bb_signal(close, indicators.get("bb_upper"), indicators.get("bb_lower")),
            "sma_signal": self._sma_signal(close, indicators.get("sma_20"), indicators.get("sma_50")),
            "stoch_signal": self._stoch_signal(indicators.get("stoch_k"), indicators.get("stoch_d")),
        }

        bull = sum(1 for v in sigs.values() if v == 1)
        bear = sum(1 for v in sigs.values() if v == -1)
        sigs["bull_count"] = bull
        sigs["bear_count"] = bear

        if bull > bear:
            composite, direction = 1, "UP"
        elif bear > bull:
            composite, direction = -1, "DOWN"
        else:
            composite, direction = 0, "NEUTRAL"

        sigs["composite"] = composite
        sigs["direction"] = direction
        return sigs

    def generate_from_df(self, df: pd.DataFrame) -> dict:
        """Compute all indicators on *df* then generate signals from the last row."""
        enriched = TA.add_all_indicators(df.copy())
        if enriched.empty:
            return {}
        last = enriched.iloc[-1].to_dict()
        return self.generate_signals(last)
