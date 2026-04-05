"""Feature engineering: derive ML features from OHLCV + indicator DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.technical import TechnicalAnalysis
from src.utils.logger import get_logger

logger = get_logger(__name__)

TA = TechnicalAnalysis()

_FEATURE_COLS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "atr",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    # Derived
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_bb_mid",
    "bb_width",
    "macd_hist_slope",
    "rsi_slope",
    "price_change_1",
    "price_change_5",
    "volume_change_1",
    "higher_high",
    "higher_low",
]


class FeatureEngineer:
    """Transform raw price data into model-ready features."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with all ML features computed."""
        df = TA.add_all_indicators(df.copy())
        close = df["close"]
        volume = df["volume"]

        df["close_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
        df["close_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
        df["close_vs_bb_mid"] = (close - df["bb_mid"]) / df["bb_mid"].replace(0, np.nan)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["macd_hist_slope"] = df["macd_hist"].diff()
        df["rsi_slope"] = df["rsi"].diff()
        df["price_change_1"] = close.pct_change(1)
        df["price_change_5"] = close.pct_change(5)
        df["volume_change_1"] = volume.pct_change(1)

        df["higher_high"] = df["higher_high"].astype(float)
        df["higher_low"] = df["higher_low"].astype(float)

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the model-input columns, dropping rows with NaNs."""
        df = self.build_features(df)
        available = [c for c in _FEATURE_COLS if c in df.columns]
        X = df[available].dropna()
        return X

    def create_labels(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.Series:
        """Create binary labels: 1 = price up ≥ threshold over *horizon* bars, 0 = price change < threshold (including downward movements)."""
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        return (future_return >= threshold).astype(int)
