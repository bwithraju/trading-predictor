"""Volatility tier definitions and classification logic.

Safety Tiers:
    Tier 1 - EXTREME VOLATILITY  → max leverage 1:1
    Tier 2 - HIGH VOLATILITY     → max leverage 2:1
    Tier 3 - NORMAL VOLATILITY   → max leverage 4:1
    Tier 4 - LOW VOLATILITY      → max leverage 8:1
    Tier 5 - EXTREME LOW         → max leverage 15:1
"""

from dataclasses import dataclass
from enum import Enum


class VolatilityTier(str, Enum):
    """Volatility classification tiers."""

    EXTREME = "EXTREME"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    EXTREME_LOW = "EXTREME_LOW"


@dataclass
class TierInfo:
    """Full metadata for a volatility tier."""

    tier: VolatilityTier
    volatility_ratio_min: float
    volatility_ratio_max: float
    max_leverage: int
    liquidation_risk_range: tuple  # (min_pct, max_pct)
    description: str
    risk_label: str
    profit_potential: str

    @property
    def recommended_leverage(self) -> int:
        """Conservative recommended leverage for this tier."""
        return self.max_leverage


# Ordered from most to least volatile
TIER_REGISTRY: list[TierInfo] = [
    TierInfo(
        tier=VolatilityTier.EXTREME,
        volatility_ratio_min=1.5,
        volatility_ratio_max=float("inf"),
        max_leverage=1,
        liquidation_risk_range=(40, 100),
        description="Extreme volatility – unpredictable price swings",
        risk_label="EXTREME RISK",
        profit_potential="Limited",
    ),
    TierInfo(
        tier=VolatilityTier.HIGH,
        volatility_ratio_min=1.0,
        volatility_ratio_max=1.5,
        max_leverage=2,
        liquidation_risk_range=(20, 30),
        description="High volatility – frequent large moves",
        risk_label="HIGH RISK",
        profit_potential="Good",
    ),
    TierInfo(
        tier=VolatilityTier.NORMAL,
        volatility_ratio_min=0.7,
        volatility_ratio_max=1.0,
        max_leverage=4,
        liquidation_risk_range=(10, 15),
        description="Normal volatility – predictable patterns",
        risk_label="MODERATE RISK",
        profit_potential="Excellent",
    ),
    TierInfo(
        tier=VolatilityTier.LOW,
        volatility_ratio_min=0.5,
        volatility_ratio_max=0.7,
        max_leverage=8,
        liquidation_risk_range=(5, 10),
        description="Low volatility – stable trending market",
        risk_label="LOW RISK",
        profit_potential="Very Good",
    ),
    TierInfo(
        tier=VolatilityTier.EXTREME_LOW,
        volatility_ratio_min=0.0,
        volatility_ratio_max=0.5,
        max_leverage=15,
        liquidation_risk_range=(2, 5),
        description="Extreme low volatility – very stable conditions",
        risk_label="LOWEST RISK",
        profit_potential="Extreme",
    ),
]

# Quick lookup by tier name
TIER_MAP: dict[VolatilityTier, TierInfo] = {t.tier: t for t in TIER_REGISTRY}


def classify_volatility_tier(volatility_ratio: float) -> TierInfo:
    """Return the :class:`TierInfo` for the given *volatility_ratio*.

    Args:
        volatility_ratio: Current ATR / historical ATR (or equivalent).

    Returns:
        The matching :class:`TierInfo` object.
    """
    if volatility_ratio <= 0:
        raise ValueError(f"volatility_ratio must be positive, got {volatility_ratio}")

    for tier_info in TIER_REGISTRY:
        if volatility_ratio >= tier_info.volatility_ratio_min:
            return tier_info

    # Fallback – should never reach here with a valid positive ratio
    return TIER_REGISTRY[-1]


def get_tier_info(tier: VolatilityTier) -> TierInfo:
    """Return :class:`TierInfo` by tier enum value."""
    return TIER_MAP[tier]
