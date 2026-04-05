"""Virtual account for paper (simulated) trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PaperPosition:
    symbol: str
    qty: float
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - self.current_price) / self.entry_price * 100


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    qty: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    outcome: str  # 'WIN', 'LOSS'
    exit_reason: str  # 'SL_HIT', 'TP_HIT', 'MANUAL', 'SIGNAL'


class PaperAccount:
    """Simulated trading account with virtual capital."""

    def __init__(
        self,
        initial_capital: float = None,
        max_open_positions: int = None,
        commission_rate: float = None,
    ):
        self.initial_capital = initial_capital or config.paper.INITIAL_CAPITAL
        self.max_open_positions = max_open_positions or config.paper.MAX_OPEN_POSITIONS
        self.commission_rate = commission_rate or config.paper.COMMISSION_RATE

        self.cash = self.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[ClosedTrade] = []

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized

    @property
    def portfolio_value(self) -> float:
        position_value = sum(
            abs(p.qty * p.current_price) for p in self.positions.values()
        )
        return self.cash + position_value

    def get_summary(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 2),
            "equity": round(self.equity, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "total_pnl": round(self.equity - self.initial_capital, 2),
            "total_pnl_pct": round(
                (self.equity - self.initial_capital) / self.initial_capital * 100, 2
            ),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
        }

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """Open a paper position if resources and limits allow."""
        if symbol in self.positions:
            logger.warning("Position already open for %s", symbol)
            return False

        if len(self.positions) >= self.max_open_positions:
            logger.warning(
                "Max open positions (%d) reached – cannot open %s",
                self.max_open_positions, symbol,
            )
            return False

        cost = qty * entry_price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            logger.warning(
                "Insufficient cash (%.2f) to open %s position costing %.2f",
                self.cash, symbol, total_cost,
            )
            return False

        self.cash -= total_cost
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            qty=qty,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
        )
        logger.info(
            "Opened paper %s position: %s qty=%s @ %s", direction, symbol, qty, entry_price
        )
        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
    ) -> Optional[ClosedTrade]:
        """Close an open position and record the trade."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            logger.warning("No open position found for %s", symbol)
            return None

        proceeds = pos.qty * exit_price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        self.cash += net_proceeds

        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.qty
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        outcome = "WIN" if pnl > 0 else "LOSS"
        trade = ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            entry_time=pos.entry_time,
            exit_time=datetime.utcnow(),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            outcome=outcome,
            exit_reason=exit_reason,
        )
        self.closed_trades.append(trade)
        logger.info(
            "Closed paper position: %s %s → %s pnl=%.2f (%.2f%%)",
            symbol, exit_reason, outcome, pnl, pnl_pct,
        )
        return trade

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def reset(self) -> None:
        """Reset the account to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        logger.info("Paper account reset to initial capital %.2f", self.initial_capital)
