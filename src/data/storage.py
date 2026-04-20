"""SQLite-backed persistent storage for OHLCV data."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db.models import SessionLocal, engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TABLE_PREFIX = "ohlcv_"


class DataStorage:
    """Store and retrieve OHLCV DataFrames in the SQLite database."""

    @staticmethod
    def _table_name(symbol: str, asset_type: str) -> str:
        safe = symbol.replace("/", "_").replace("-", "_").lower()
        return f"{_TABLE_PREFIX}{asset_type}_{safe}"

    def save(self, df: pd.DataFrame, symbol: str, asset_type: str) -> None:
        """Persist *df* to the database table for *symbol*."""
        if df.empty:
            logger.warning("Attempted to save empty DataFrame for %s", symbol)
            return
        table = self._table_name(symbol, asset_type)
        try:
            df.to_sql(table, con=engine, if_exists="replace", index=True)
            logger.info("Saved %d rows for %s to table '%s'", len(df), symbol, table)
        except Exception as exc:
            logger.error("Failed to save data for %s: %s", symbol, exc)
            raise

    def load(self, symbol: str, asset_type: str) -> pd.DataFrame:
        """Load OHLCV data for *symbol* from the database."""
        table = self._table_name(symbol, asset_type)
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                    {"t": table},
                )
                if not result.fetchone():
                    logger.info("No stored data found for %s", symbol)
                    return pd.DataFrame()
            df = pd.read_sql_table(table, con=engine, index_col="timestamp")
            df.index = pd.to_datetime(df.index)
            logger.info("Loaded %d rows for %s from table '%s'", len(df), symbol, table)
            return df
        except Exception as exc:
            logger.error("Failed to load data for %s: %s", symbol, exc)
            return pd.DataFrame()
