"""Leverage and position-size calculations.

Provides:
  - :class:`LeverageCalculator` – core financial math for leverage and position sizing.
"""

from __future__ import annotations

import math


class LeverageCalculator:
    """Core financial math for leverage, position sizing, and liquidation risk.

    All monetary inputs / outputs are in the same currency unit (e.g. USD).
    """

    # Hard cap on any single recommendation
    MAX_ABSOLUTE_LEVERAGE = 20

    def buying_power(self, account_size: float, leverage: int) -> float:
        """Total buying power at a given leverage.

        Args:
            account_size: Current account equity in USD.
            leverage: Leverage multiplier (e.g. 4 → 4:1).

        Returns:
            Total buying power in USD.
        """
        if account_size <= 0 or leverage <= 0:
            raise ValueError("account_size and leverage must be positive")
        return account_size * leverage

    def max_position_size(
        self,
        account_size: float,
        risk_percent: float,
        stop_loss_percent: float,
    ) -> float:
        """Maximum position size based on fixed-percentage risk.

        The position is sized so that hitting the stop-loss costs at most
        ``risk_percent`` of ``account_size``.

        Args:
            account_size: Account equity in USD.
            risk_percent: Risk per trade as a percentage (e.g. 2.0 = 2 %).
            stop_loss_percent: Distance to stop-loss as a percentage of entry
                price (e.g. 5.0 = 5 %).

        Returns:
            Maximum position size in USD.
        """
        if account_size <= 0:
            raise ValueError("account_size must be positive")
        if risk_percent <= 0 or risk_percent > 100:
            raise ValueError("risk_percent must be in (0, 100]")
        if stop_loss_percent <= 0 or stop_loss_percent > 100:
            raise ValueError("stop_loss_percent must be in (0, 100]")

        risk_amount = account_size * (risk_percent / 100)
        position = risk_amount / (stop_loss_percent / 100)
        return position

    def max_leverage_for_position(self, position_size: float, account_size: float) -> float:
        """Leverage implied by a given position size and account equity."""
        if account_size <= 0:
            raise ValueError("account_size must be positive")
        return position_size / account_size

    def liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: str = "long",
        maintenance_margin_rate: float = 0.005,
    ) -> float:
        """Approximate liquidation price for a leveraged position.

        Args:
            entry_price: Entry price.
            leverage: Leverage multiplier.
            direction: ``"long"`` or ``"short"``.
            maintenance_margin_rate: Maintenance margin as a fraction
                (e.g. 0.005 = 0.5 %).

        Returns:
            Liquidation price.
        """
        if entry_price <= 0 or leverage <= 0:
            raise ValueError("entry_price and leverage must be positive")
        if direction not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")

        initial_margin = 1 / leverage
        if direction == "long":
            return entry_price * (1 - initial_margin + maintenance_margin_rate)
        else:
            return entry_price * (1 + initial_margin - maintenance_margin_rate)

    def liquidation_risk(
        self,
        volatility: float,
        trend_strength: float,
        leverage: int,
    ) -> float:
        """Estimate liquidation risk (0–100 %) for a position.

        Uses a simplified model:
            max_adverse_move ≈ volatility × (2 – trend_strength)
            margin = 1 / leverage
            risk = clamp((max_adverse_move / margin) × 100, 0, 100)

        Args:
            volatility: Historical volatility as a fraction (e.g. 0.30 = 30 %).
            trend_strength: ADX-normalised trend strength in [0, 1].
            leverage: Leverage multiplier.

        Returns:
            Estimated liquidation risk in percent.
        """
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        if not (0 <= trend_strength <= 1):
            raise ValueError("trend_strength must be in [0, 1]")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        max_adverse_move = volatility * (2.0 - trend_strength)
        margin = 1.0 / leverage
        risk = (max_adverse_move / margin) * 100
        return float(min(max(risk, 0.0), 100.0))

    def risk_level_label(self, liquidation_risk_pct: float) -> str:
        """Map a liquidation risk percentage to a human-readable label."""
        if liquidation_risk_pct > 30:
            return "CRITICAL"
        if liquidation_risk_pct > 15:
            return "HIGH"
        if liquidation_risk_pct > 5:
            return "MODERATE"
        return "LOW"

    def daily_stop_loss_amount(self, account_size: float, daily_loss_limit_pct: float = 5.0) -> float:
        """Dollar amount corresponding to the daily stop-loss limit.

        Args:
            account_size: Account equity in USD.
            daily_loss_limit_pct: Maximum daily loss as a percentage of equity.

        Returns:
            Maximum daily loss amount in USD.
        """
        if account_size <= 0:
            raise ValueError("account_size must be positive")
        return account_size * (daily_loss_limit_pct / 100)

    def recommended_leverage_from_risk(
        self,
        account_size: float,
        risk_percent: float,
        stop_loss_percent: float,
        model_recommendation: int,
    ) -> int:
        """Combine position-sizing math with model recommendation.

        Returns the *minimum* of the model recommendation and the leverage
        implied by the maximum safe position size.

        Args:
            account_size: Account equity in USD.
            risk_percent: Risk per trade (%).
            stop_loss_percent: Stop-loss distance (%).
            model_recommendation: Leverage from the ML model.

        Returns:
            Final recommended leverage (integer, min 1).
        """
        max_pos = self.max_position_size(account_size, risk_percent, stop_loss_percent)
        max_lev = self.max_leverage_for_position(max_pos, account_size)
        final = int(min(max_lev, model_recommendation, self.MAX_ABSOLUTE_LEVERAGE))
        return max(final, 1)

    def account_summary(
        self,
        account_size: float,
        leverage: int,
        risk_percent: float = 2.0,
        stop_loss_percent: float = 5.0,
        daily_loss_limit_pct: float = 5.0,
    ) -> dict:
        """Compute a full account summary at the given leverage.

        Returns:
            Dictionary with buying_power, max_position_size,
            daily_stop_loss, leverage, account_size.
        """
        return {
            "account_size": account_size,
            "leverage": leverage,
            "buying_power": self.buying_power(account_size, leverage),
            "max_position_size": self.max_position_size(account_size, risk_percent, stop_loss_percent),
            "daily_stop_loss": self.daily_stop_loss_amount(account_size, daily_loss_limit_pct),
        }
