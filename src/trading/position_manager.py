"""Open position tracking and management via Alpaca."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.trading.alpaca_client import AlpacaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    side: str          # 'long' or 'short'
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


class PositionManager:
    """Track and manage open positions via the Alpaca API."""

    def __init__(self, client: AlpacaClient):
        self.client = client

    def get_all_positions(self) -> List[Position]:
        """Return all open positions."""
        if not self.client.is_connected:
            return []
        try:
            raw = self.client.api.list_positions()
            return [self._to_position(p) for p in raw]
        except Exception as exc:
            logger.error("Error fetching positions: %s", exc)
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for *symbol*, or None."""
        if not self.client.is_connected:
            return None
        try:
            p = self.client.api.get_position(symbol)
            return self._to_position(p)
        except Exception as exc:
            logger.debug("No position found for %s: %s", symbol, exc)
            return None

    def close_position(self, symbol: str) -> bool:
        """Close the entire position for *symbol* at market."""
        if not self.client.is_connected:
            logger.warning("Alpaca not connected – cannot close position for %s", symbol)
            return False
        try:
            self.client.api.close_position(symbol)
            logger.info("Position closed for %s", symbol)
            return True
        except Exception as exc:
            logger.error("Error closing position for %s: %s", symbol, exc)
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        if not self.client.is_connected:
            return False
        try:
            self.client.api.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as exc:
            logger.error("Error closing all positions: %s", exc)
            return False

    def get_position_summary(self) -> Dict[str, Any]:
        """Return a summary of all open positions."""
        positions = self.get_all_positions()
        total_market_value = sum(p.market_value for p in positions)
        total_unrealized_pl = sum(p.unrealized_pl for p in positions)
        return {
            "count": len(positions),
            "total_market_value": round(total_market_value, 2),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pl": round(p.unrealized_pl, 2),
                    "unrealized_plpc": round(p.unrealized_plpc * 100, 2),
                }
                for p in positions
            ],
        }

    def calculate_position_size(
        self,
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 2.0,
        max_open_positions: int = 5,
    ) -> float:
        """Calculate the number of shares to trade given risk parameters."""
        risk_amount = account_size * risk_percent / 100
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return 0.0
        raw_size = risk_amount / sl_distance
        # Limit by max open positions allocation
        max_allocation = account_size / max_open_positions / entry_price
        return min(raw_size, max_allocation)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_position(p) -> Position:
        return Position(
            symbol=p.symbol,
            qty=float(p.qty),
            side=p.side,
            entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
            unrealized_plpc=float(p.unrealized_plpc),
        )
