"""Order execution and management via Alpaca."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.trading.alpaca_client import AlpacaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrderRecord:
    order_id: str
    symbol: str
    side: str          # 'buy' or 'sell'
    order_type: str    # 'market', 'limit', 'stop', 'stop_limit'
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class OrderManager:
    """Execute and track orders via the Alpaca API."""

    def __init__(self, client: AlpacaClient):
        self.client = client
        self._order_history: List[OrderRecord] = []

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ) -> Optional[OrderRecord]:
        """Submit a market order."""
        if not self.client.is_connected:
            logger.warning("Alpaca not connected – cannot submit market order for %s", symbol)
            return None
        try:
            order = self.client.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            record = self._to_record(order)
            self._order_history.append(record)
            logger.info("Market order submitted: %s %s %s @ market", side.upper(), qty, symbol)
            return record
        except Exception as exc:
            logger.error("Market order failed for %s: %s", symbol, exc)
            return None

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
    ) -> Optional[OrderRecord]:
        """Submit a limit order."""
        if not self.client.is_connected:
            logger.warning("Alpaca not connected – cannot submit limit order for %s", symbol)
            return None
        try:
            order = self.client.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="limit",
                time_in_force="day",
                limit_price=limit_price,
            )
            record = self._to_record(order)
            self._order_history.append(record)
            logger.info(
                "Limit order submitted: %s %s %s @ %s", side.upper(), qty, symbol, limit_price
            )
            return record
        except Exception as exc:
            logger.error("Limit order failed for %s: %s", symbol, exc)
            return None

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[OrderRecord]:
        """Submit a bracket (entry + SL + TP) order."""
        if not self.client.is_connected:
            logger.warning("Alpaca not connected – cannot submit bracket order for %s", symbol)
            return None
        try:
            order = self.client.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="limit",
                time_in_force="day",
                limit_price=limit_price,
                order_class="bracket",
                stop_loss={"stop_price": stop_loss},
                take_profit={"limit_price": take_profit},
            )
            record = self._to_record(order)
            self._order_history.append(record)
            logger.info(
                "Bracket order submitted: %s %s %s | SL=%s TP=%s",
                side.upper(), qty, symbol, stop_loss, take_profit,
            )
            return record
        except Exception as exc:
            logger.error("Bracket order failed for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Order status
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest status for a given order ID."""
        if not self.client.is_connected:
            return None
        try:
            order = self.client.api.get_order(order_id)
            return self._order_to_dict(order)
        except Exception as exc:
            logger.error("Error fetching order %s: %s", order_id, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.client.is_connected:
            return False
        try:
            self.client.api.cancel_order(order_id)
            logger.info("Order %s cancelled", order_id)
            return True
        except Exception as exc:
            logger.error("Error cancelling order %s: %s", order_id, exc)
            return False

    def get_order_history(self) -> List[Dict[str, Any]]:
        """Return the local order history as a list of dicts."""
        return [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": o.side,
                "order_type": o.order_type,
                "qty": o.qty,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "status": o.status,
                "filled_qty": o.filled_qty,
                "filled_avg_price": o.filled_avg_price,
                "created_at": o.created_at.isoformat(),
            }
            for o in self._order_history
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_record(order) -> OrderRecord:
        return OrderRecord(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            qty=float(order.qty),
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
            status=order.status,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
            filled_avg_price=(
                float(order.filled_avg_price) if order.filled_avg_price else None
            ),
        )

    @staticmethod
    def _order_to_dict(order) -> Dict[str, Any]:
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "qty": float(order.qty),
            "status": order.status,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "filled_avg_price": (
                float(order.filled_avg_price) if order.filled_avg_price else None
            ),
        }
