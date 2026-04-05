"""Tests for risk calculator."""
from __future__ import annotations

import pytest

from src.risk.calculator import RiskCalculator, RiskMetrics


class TestRiskCalculator:
    def test_long_basic(self):
        calc = RiskCalculator(sl_percent=2.0, reward_ratio=2.0)
        result = calc.calculate(entry_price=100.0, direction="LONG",
                                risk_percent=2.0, account_size=10_000.0)
        assert isinstance(result, RiskMetrics)
        assert result.stop_loss < 100.0
        assert result.take_profit > 100.0
        assert result.max_loss == pytest.approx(200.0, rel=0.01)
        assert result.direction == "LONG"

    def test_short_basic(self):
        calc = RiskCalculator(sl_percent=2.0, reward_ratio=2.0)
        result = calc.calculate(entry_price=100.0, direction="SHORT",
                                risk_percent=2.0, account_size=10_000.0)
        assert result.stop_loss > 100.0
        assert result.take_profit < 100.0
        assert result.direction == "SHORT"

    def test_invalid_direction(self):
        calc = RiskCalculator()
        with pytest.raises(ValueError, match="direction"):
            calc.calculate(entry_price=100.0, direction="HOLD")

    def test_from_exit(self):
        calc = RiskCalculator(reward_ratio=2.0)
        result = calc.calculate_from_exit(entry_price=100.0, exit_price=96.0,
                                          direction="LONG", account_size=10_000.0)
        assert result.stop_loss < result.entry_price
        assert result.take_profit > result.entry_price

    def test_position_size_capped(self):
        # With very tight SL, position size would be huge, but should be capped
        calc = RiskCalculator(sl_percent=0.01, reward_ratio=2.0)
        result = calc.calculate(entry_price=100.0, direction="LONG",
                                risk_percent=2.0, account_size=10_000.0)
        max_units = 10_000.0 * 0.20 / 100.0
        assert result.position_size <= max_units + 1.0  # allow tiny float error

    def test_risk_reward_ratio(self):
        calc = RiskCalculator(sl_percent=2.0, reward_ratio=2.0)
        result = calc.calculate(entry_price=100.0, direction="LONG",
                                risk_percent=2.0, account_size=10_000.0)
        assert result.risk_reward_ratio == pytest.approx(2.0, rel=0.01)
