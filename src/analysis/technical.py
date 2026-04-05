"""Compute technical indicators on OHLCV DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalAnalysis:
    """Add all technical indicator columns to a price DataFrame."""

    # ------------------------------------------------------------------
    # Trend indicators
    # ------------------------------------------------------------------

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (macd_line, signal_line, histogram)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ------------------------------------------------------------------
    # Momentum indicators
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Return (%K, %D) stochastic oscillator series."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        range_hl = (highest_high - lowest_low).replace(0, np.nan)
        k = 100 * (close - lowest_low) / range_hl
        d = k.rolling(window=d_period).mean()
        return k, d

    # ------------------------------------------------------------------
    # Volatility indicators
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (upper_band, middle_band, lower_band)."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    # ------------------------------------------------------------------
    # Pattern helpers
    # ------------------------------------------------------------------

    @staticmethod
    def higher_highs(high: pd.Series, window: int = 5) -> pd.Series:
        """Boolean series: True when *high* is a higher high over *window* bars."""
        return high > high.shift(window)

    @staticmethod
    def higher_lows(low: pd.Series, window: int = 5) -> pd.Series:
        """Boolean series: True when *low* is a higher low over *window* bars."""
        return low > low.shift(window)

    @staticmethod
    def support_resistance(
        close: pd.Series, window: int = 20
    ) -> tuple[pd.Series, pd.Series]:
        """Return rolling (support, resistance) levels."""
        support = close.rolling(window=window).min()
        resistance = close.rolling(window=window).max()
        return support, resistance

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and append them as columns to *df*."""
        if df.empty or len(df) < 30:
            logger.warning("DataFrame too short to compute indicators (len=%d)", len(df))
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Trend
        df["sma_20"] = self.sma(close, 20)
        df["sma_50"] = self.sma(close, 50)
        df["ema_12"] = self.ema(close, 12)
        df["ema_26"] = self.ema(close, 26)
        df["macd"], df["macd_signal"], df["macd_hist"] = self.macd(close)

        # Momentum
        df["rsi"] = self.rsi(close)
        df["stoch_k"], df["stoch_d"] = self.stochastic(high, low, close)

        # Volatility
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = self.bollinger_bands(close)
        df["atr"] = self.atr(high, low, close)

        # Pattern
        df["higher_high"] = self.higher_highs(high)
        df["higher_low"] = self.higher_lows(low)
        df["support"], df["resistance"] = self.support_resistance(close)

        logger.debug("Added all indicators; DataFrame now has %d columns", len(df.columns))
        return df

    def get_current_indicators(self, df: pd.DataFrame) -> dict:
        """Return the most recent row of indicators as a plain dict."""
        df = self.add_all_indicators(df.copy())
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            col: (None if pd.isna(row[col]) else float(row[col]))
            for col in df.columns
        }
