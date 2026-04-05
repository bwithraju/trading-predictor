"""Filesystem-based cache for OHLCV DataFrames (CSV files)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCache:
    """Cache OHLCV data as CSV files on disk."""

    def __init__(self, cache_dir: str = None, ttl: int = None):
        self.cache_dir = Path(cache_dir or config.data.CACHE_DIR)
        self.ttl = ttl or config.data.CACHE_TTL_SECONDS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, asset_type: str) -> Path:
        safe = symbol.replace("/", "_").replace("-", "_").lower()
        return self.cache_dir / f"{asset_type}_{safe}.csv"

    def is_valid(self, symbol: str, asset_type: str) -> bool:
        """Return True when the cache file exists and is within the TTL window."""
        path = self._path(symbol, asset_type)
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.ttl

    def save(self, df: pd.DataFrame, symbol: str, asset_type: str) -> None:
        if df.empty:
            return
        path = self._path(symbol, asset_type)
        df.to_csv(path)
        logger.debug("Cached %d rows for %s at %s", len(df), symbol, path)

    def load(self, symbol: str, asset_type: str) -> pd.DataFrame:
        path = self._path(symbol, asset_type)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.debug("Loaded %d rows for %s from cache", len(df), symbol)
            return df
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def invalidate(self, symbol: str, asset_type: str) -> None:
        path = self._path(symbol, asset_type)
        if path.exists():
            os.remove(path)
            logger.debug("Invalidated cache for %s", symbol)
