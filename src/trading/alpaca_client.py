"""Alpaca REST API client wrapper."""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaClient:
    """Thin wrapper around the Alpaca trading API.

    Uses ``alpaca-trade-api`` when credentials are configured, otherwise
    operates in a *mock/paper-only* mode so that unit tests can import this
    module without real credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
    ):
        self.api_key = api_key or config.alpaca.API_KEY
        self.secret_key = secret_key or config.alpaca.SECRET_KEY
        self.base_url = base_url or config.alpaca.BASE_URL
        self.paper = paper if paper is not None else config.alpaca.PAPER_MODE
        self._api = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the Alpaca API connection.

        Returns True on success, False if credentials are missing/invalid.
        """
        if not self.api_key or not self.secret_key:
            logger.warning(
                "Alpaca credentials not configured – running in mock mode"
            )
            return False
        try:
            import alpaca_trade_api as tradeapi  # type: ignore

            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version="v2",
            )
            # Validate by fetching account
            self._api.get_account()
            mode = "paper" if self.paper else "live"
            logger.info("Connected to Alpaca API (%s mode)", mode)
            return True
        except Exception as exc:
            logger.error("Failed to connect to Alpaca API: %s", exc)
            return False

    @property
    def is_connected(self) -> bool:
        return self._api is not None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Return account information as a plain dict."""
        if not self.is_connected:
            return {}
        try:
            acct = self._api.get_account()
            return {
                "id": acct.id,
                "status": acct.status,
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "currency": acct.currency,
                "paper": self.paper,
            }
        except Exception as exc:
            logger.error("Error fetching account: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Return the latest quote for *symbol*."""
        if not self.is_connected:
            return {}
        try:
            quote = self._api.get_latest_quote(symbol)
            return {
                "symbol": symbol,
                "ask_price": float(quote.ap),
                "bid_price": float(quote.bp),
                "ask_size": float(quote.as_),
                "bid_size": float(quote.bs),
            }
        except Exception as exc:
            logger.error("Error fetching quote for %s: %s", symbol, exc)
            return {}

    # ------------------------------------------------------------------
    # Raw API access (for OrderManager / PositionManager)
    # ------------------------------------------------------------------

    @property
    def api(self):
        """Expose the raw alpaca_trade_api.REST object."""
        return self._api
