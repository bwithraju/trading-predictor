"""Account information and portfolio tracking via Alpaca."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.trading.alpaca_client import AlpacaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    status: str
    paper: bool


class AccountManager:
    """Retrieve account info and monitor daily P&L and risk limits."""

    def __init__(
        self,
        client: AlpacaClient,
        daily_loss_limit_pct: float = 5.0,
    ):
        self.client = client
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._start_of_day_equity: Optional[float] = None

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Optional[AccountSnapshot]:
        """Return a current account snapshot."""
        if not self.client.is_connected:
            return None
        try:
            raw = self.client.get_account()
            if not raw:
                return None

            equity = raw.get("equity", 0.0)
            if self._start_of_day_equity is None:
                self._start_of_day_equity = equity

            daily_pnl = equity - self._start_of_day_equity
            daily_pnl_pct = (
                daily_pnl / self._start_of_day_equity * 100
                if self._start_of_day_equity > 0
                else 0.0
            )

            return AccountSnapshot(
                equity=equity,
                cash=raw.get("cash", 0.0),
                buying_power=raw.get("buying_power", 0.0),
                portfolio_value=raw.get("portfolio_value", 0.0),
                daily_pnl=round(daily_pnl, 2),
                daily_pnl_pct=round(daily_pnl_pct, 2),
                status=raw.get("status", "unknown"),
                paper=raw.get("paper", True),
            )
        except Exception as exc:
            logger.error("Error fetching account snapshot: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Risk monitoring
    # ------------------------------------------------------------------

    def is_daily_loss_limit_breached(self) -> bool:
        """Return True if daily loss exceeds the configured limit."""
        snap = self.get_snapshot()
        if snap is None:
            return False
        if snap.daily_pnl_pct < -self.daily_loss_limit_pct:
            logger.warning(
                "Daily loss limit breached: %.2f%% (limit: %.2f%%)",
                snap.daily_pnl_pct,
                self.daily_loss_limit_pct,
            )
            return True
        return False

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Return a portfolio summary dict."""
        snap = self.get_snapshot()
        if snap is None:
            return {"error": "Not connected to Alpaca"}
        return {
            "equity": snap.equity,
            "cash": snap.cash,
            "buying_power": snap.buying_power,
            "portfolio_value": snap.portfolio_value,
            "daily_pnl": snap.daily_pnl,
            "daily_pnl_pct": snap.daily_pnl_pct,
            "daily_loss_limit_breached": self.is_daily_loss_limit_breached(),
            "status": snap.status,
            "paper_mode": snap.paper,
        }
