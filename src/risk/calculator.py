"""SL / TP / ML and position-size calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import config
from src.utils.logger import get_logger
from src.utils.validators import validate_price, validate_risk_percent

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    entry_price: float
    stop_loss: float
    take_profit: float
    max_loss: float
    position_size: float
    risk_percent: float
    reward_ratio: float
    risk_reward_ratio: float
    direction: str  # 'LONG' or 'SHORT'


class RiskCalculator:
    """Calculate stop-loss, take-profit, max-loss, and position size."""

    def __init__(
        self,
        sl_percent: float = None,
        reward_ratio: float = None,
    ):
        self.sl_percent = sl_percent or config.risk.SL_PERCENT_LONG
        self.reward_ratio = reward_ratio or config.risk.DEFAULT_REWARD_RATIO

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        entry_price: float,
        direction: str = "LONG",
        risk_percent: float = None,
        account_size: float = 10_000.0,
        sl_percent: float = None,
        reward_ratio: float = None,
    ) -> RiskMetrics:
        """Compute a full set of risk metrics for one trade.

        Parameters
        ----------
        entry_price:  Entry price for the trade.
        direction:    ``"LONG"`` or ``"SHORT"``.
        risk_percent: Percentage of *account_size* risked on this trade.
        account_size: Total account value in currency units.
        sl_percent:   Stop-loss distance as a percentage of *entry_price*.
        reward_ratio: Take-profit multiplier relative to stop-loss distance.
        """
        entry_price = validate_price(entry_price, "entry_price")
        risk_percent = validate_risk_percent(risk_percent or config.risk.DEFAULT_RISK_PERCENT)
        sl_pct = sl_percent or self.sl_percent
        rr = reward_ratio or self.reward_ratio
        direction = direction.upper()

        if direction == "LONG":
            stop_loss = entry_price * (1 - sl_pct / 100)
            take_profit = entry_price * (1 + sl_pct / 100 * rr)
        elif direction == "SHORT":
            stop_loss = entry_price * (1 + sl_pct / 100)
            take_profit = entry_price * (1 - sl_pct / 100 * rr)
        else:
            raise ValueError(f"direction must be 'LONG' or 'SHORT', got '{direction}'")

        max_loss = account_size * risk_percent / 100
        sl_distance = abs(entry_price - stop_loss)
        position_size = max_loss / sl_distance if sl_distance > 0 else 0

        # Cap position size to MAX_POSITION_SIZE of account
        max_units = account_size * config.risk.MAX_POSITION_SIZE / entry_price
        position_size = min(position_size, max_units)

        tp_distance = abs(take_profit - entry_price)
        risk_reward = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0.0

        return RiskMetrics(
            entry_price=round(entry_price, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            max_loss=round(max_loss, 2),
            position_size=round(position_size, 4),
            risk_percent=risk_percent,
            reward_ratio=rr,
            risk_reward_ratio=risk_reward,
            direction=direction,
        )

    # ------------------------------------------------------------------
    # Convenience: calculate from explicit exit price
    # ------------------------------------------------------------------

    def calculate_from_exit(
        self,
        entry_price: float,
        exit_price: float,
        direction: str = "LONG",
        account_size: float = 10_000.0,
        risk_percent: float = None,
    ) -> RiskMetrics:
        """Back-calculate SL/TP distances when the exit price is known."""
        entry_price = validate_price(entry_price, "entry_price")
        exit_price = validate_price(exit_price, "exit_price")
        risk_percent = validate_risk_percent(risk_percent or config.risk.DEFAULT_RISK_PERCENT)
        direction = direction.upper()

        sl_distance = abs(entry_price - exit_price)
        sl_pct = sl_distance / entry_price * 100
        rr = self.reward_ratio
        return self.calculate(
            entry_price=entry_price,
            direction=direction,
            risk_percent=risk_percent,
            account_size=account_size,
            sl_percent=sl_pct,
            reward_ratio=rr,
        )
