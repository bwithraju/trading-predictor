"""Real-time risk monitoring for live and paper trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskStatus:
    daily_pnl_pct: float
    open_positions: int
    daily_loss_limit_breached: bool
    max_positions_reached: bool
    can_trade: bool
    warnings: List[str]


class RiskMonitor:
    """Monitor risk limits during live/paper trading sessions."""

    def __init__(
        self,
        max_risk_per_trade_pct: float = None,
        daily_loss_limit_pct: float = None,
        max_open_positions: int = None,
    ):
        self.max_risk_per_trade_pct = (
            max_risk_per_trade_pct or config.live.MAX_RISK_PER_TRADE_PCT
        )
        self.daily_loss_limit_pct = (
            daily_loss_limit_pct or config.live.DAILY_LOSS_LIMIT_PCT
        )
        self.max_open_positions = max_open_positions or config.live.MAX_OPEN_POSITIONS

    def check_trade_allowed(
        self,
        current_equity: float,
        start_of_day_equity: float,
        open_position_count: int,
        proposed_trade_size_pct: float,
    ) -> RiskStatus:
        """Evaluate whether a new trade is allowed given current risk state."""
        warnings = []

        # Daily P&L
        daily_pnl_pct = 0.0
        if start_of_day_equity > 0:
            daily_pnl_pct = (current_equity - start_of_day_equity) / start_of_day_equity * 100

        daily_breach = daily_pnl_pct < -self.daily_loss_limit_pct
        if daily_breach:
            warnings.append(
                f"Daily loss limit breached: {daily_pnl_pct:.2f}% < "
                f"-{self.daily_loss_limit_pct:.2f}%"
            )

        # Position count
        max_pos_reached = open_position_count >= self.max_open_positions
        if max_pos_reached:
            warnings.append(
                f"Max open positions reached: {open_position_count}/{self.max_open_positions}"
            )

        # Trade size
        if proposed_trade_size_pct > self.max_risk_per_trade_pct:
            warnings.append(
                f"Proposed trade risk {proposed_trade_size_pct:.2f}% > "
                f"max {self.max_risk_per_trade_pct:.2f}%"
            )

        can_trade = not daily_breach and not max_pos_reached
        if proposed_trade_size_pct > self.max_risk_per_trade_pct:
            can_trade = False

        return RiskStatus(
            daily_pnl_pct=round(daily_pnl_pct, 2),
            open_positions=open_position_count,
            daily_loss_limit_breached=daily_breach,
            max_positions_reached=max_pos_reached,
            can_trade=can_trade,
            warnings=warnings,
        )

    def validate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
    ) -> Dict[str, float]:
        """Calculate and validate position size for a trade."""
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return {"qty": 0.0, "risk_pct": 0.0, "risk_amount": 0.0}

        max_risk_amount = account_equity * self.max_risk_per_trade_pct / 100
        qty = max_risk_amount / sl_distance
        actual_risk_pct = (qty * sl_distance) / account_equity * 100

        return {
            "qty": round(qty, 4),
            "risk_pct": round(actual_risk_pct, 4),
            "risk_amount": round(max_risk_amount, 2),
        }
