"""Fetch OHLCV data from yfinance (stocks) and Binance via CCXT (crypto)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from config import config
from src.utils.logger import get_logger
from src.utils.validators import validate_symbol

logger = get_logger(__name__)


class DataFetcher:
    """Unified data fetcher for stocks (yfinance) and crypto (CCXT/Binance)."""

    def __init__(self):
        self._exchange = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exchange(self):
        """Lazily initialise the CCXT Binance exchange."""
        if self._exchange is None:
            try:
                import ccxt  # type: ignore

                exchange_cls = getattr(ccxt, config.data.DEFAULT_CRYPTO_EXCHANGE)
                self._exchange = exchange_cls(
                    {
                        "apiKey": config.data.BINANCE_API_KEY or None,
                        "secret": config.data.BINANCE_SECRET or None,
                        "enableRateLimit": True,
                    }
                )
            except Exception as exc:
                logger.error("Failed to initialise CCXT exchange: %s", exc)
                raise
        return self._exchange

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = None,
        interval: str = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a stock symbol via yfinance."""
        symbol = validate_symbol(symbol)
        interval = interval or config.data.DEFAULT_STOCK_INTERVAL
        period = period or config.data.DEFAULT_STOCK_PERIOD

        logger.info("Fetching stock data for %s (interval=%s)", symbol, interval)
        try:
            ticker = yf.Ticker(symbol)
            if start:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning("No data returned for stock %s", symbol)
                return pd.DataFrame()

            df = self._normalize_columns(df)
            df = self._handle_missing(df)
            logger.info("Fetched %d rows for %s", len(df), symbol)
            return df
        except Exception as exc:
            logger.error("Error fetching stock data for %s: %s", symbol, exc)
            raise

    def fetch_crypto_data(
        self,
        symbol: str,
        timeframe: str = None,
        limit: int = None,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a crypto pair via CCXT/Binance."""
        symbol = validate_symbol(symbol)
        timeframe = timeframe or config.data.DEFAULT_CRYPTO_TIMEFRAME
        limit = limit or config.data.DEFAULT_CRYPTO_LIMIT

        logger.info("Fetching crypto data for %s (timeframe=%s)", symbol, timeframe)
        try:
            exchange = self._get_exchange()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            if not ohlcv:
                logger.warning("No data returned for crypto %s", symbol)
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = self._handle_missing(df)
            logger.info("Fetched %d rows for %s", len(df), symbol)
            return df
        except Exception as exc:
            logger.error("Error fetching crypto data for %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names and keep only OHLCV columns."""
        df.columns = [c.lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then back-fill missing values and drop remaining NaNs."""
        df = df.ffill().bfill()
        df.dropna(inplace=True)
        return df
