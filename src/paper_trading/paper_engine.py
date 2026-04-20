"""Paper trading engine: process signals and manage simulated order execution."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.paper_trading.paper_account import PaperAccount, ClosedTrade
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaperEngine:
    """Execute simulated trades based on prediction signals."""

    def __init__(self, account: Optional[PaperAccount] = None):
        self.account = account or PaperAccount()
        self._is_running = False

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._is_running = True
        logger.info(
            "Paper trading engine started (capital=%.2f)", self.account.initial_capital
        )

    def stop(self) -> None:
        self._is_running = False
        logger.info("Paper trading engine stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_signal(
        self,
        symbol: str,
        direction: str,       # 'LONG' or 'SHORT'
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,  # number of shares
    ) -> bool:
        """Open a paper position for an incoming trading signal."""
        if not self._is_running:
            logger.warning("Paper engine is not running – ignoring signal for %s", symbol)
            return False

        return self.account.open_position(
            symbol=symbol,
            direction=direction,
            qty=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def process_bar(self, symbol: str, high: float, low: float, close: float) -> Optional[str]:
        """Check if SL or TP is triggered for *symbol* given bar data.

        Returns the exit_reason string if a position was closed, else None.
        """
        pos = self.account.positions.get(symbol)
        if pos is None:
            return None

        pos.current_price = close

        if pos.direction == "LONG":
            if low <= pos.stop_loss:
                self.account.close_position(symbol, pos.stop_loss, "SL_HIT")
                return "SL_HIT"
            if high >= pos.take_profit:
                self.account.close_position(symbol, pos.take_profit, "TP_HIT")
                return "TP_HIT"
        else:  # SHORT
            if high >= pos.stop_loss:
                self.account.close_position(symbol, pos.stop_loss, "SL_HIT")
                return "SL_HIT"
            if low <= pos.take_profit:
                self.account.close_position(symbol, pos.take_profit, "TP_HIT")
                return "TP_HIT"

        return None

    def process_dataframe(self, symbol: str, df: pd.DataFrame) -> List[str]:
        """Feed a DataFrame of OHLCV bars through the engine for *symbol*.

        Returns a list of exit_reason strings for each bar that triggered a close.
        """
        exits = []
        for _, row in df.iterrows():
            result = self.process_bar(
                symbol=symbol,
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            if result:
                exits.append(result)
        return exits

    def close_position(self, symbol: str, exit_price: float) -> Optional[ClosedTrade]:
        """Manually close a paper position."""
        return self.account.close_position(symbol, exit_price, "MANUAL")

    def get_status(self) -> dict:
        return {
            "running": self._is_running,
            "account": self.account.get_summary(),
        }

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        self.account.update_prices(prices)
