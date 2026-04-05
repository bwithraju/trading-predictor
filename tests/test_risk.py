"""Unit tests for src/risk/."""
import pytest

from src.risk.calculator import RiskCalculator


class TestRiskCalculator:
    calc = RiskCalculator(sl_percent=2.0, reward_ratio=2.0)

    def test_long_stop_loss_below_entry(self):
        risk = self.calc.calculate(100.0, direction="LONG")
        assert risk.stop_loss < risk.entry_price

    def test_long_take_profit_above_entry(self):
        risk = self.calc.calculate(100.0, direction="LONG")
        assert risk.take_profit > risk.entry_price

    def test_short_stop_loss_above_entry(self):
        risk = self.calc.calculate(100.0, direction="SHORT")
        assert risk.stop_loss > risk.entry_price

    def test_short_take_profit_below_entry(self):
        risk = self.calc.calculate(100.0, direction="SHORT")
        assert risk.take_profit < risk.entry_price

    def test_sl_2_percent_long(self):
        risk = self.calc.calculate(100.0, direction="LONG", sl_percent=2.0)
        assert abs(risk.stop_loss - 98.0) < 0.01

    def test_tp_4_percent_long_2x_rr(self):
        risk = self.calc.calculate(100.0, direction="LONG", sl_percent=2.0, reward_ratio=2.0)
        assert abs(risk.take_profit - 104.0) < 0.01

    def test_max_loss_calculation(self):
        risk = self.calc.calculate(100.0, direction="LONG", risk_percent=2.0, account_size=10_000)
        assert abs(risk.max_loss - 200.0) < 0.01

    def test_position_size_positive(self):
        risk = self.calc.calculate(100.0, direction="LONG", risk_percent=2.0, account_size=10_000)
        assert risk.position_size > 0

    def test_risk_reward_ratio(self):
        risk = self.calc.calculate(100.0, direction="LONG", sl_percent=2.0, reward_ratio=2.0)
        assert abs(risk.risk_reward_ratio - 2.0) < 0.01

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            self.calc.calculate(100.0, direction="SIDEWAYS")

    def test_invalid_entry_price_raises(self):
        with pytest.raises(ValueError):
            self.calc.calculate(-10.0, direction="LONG")

    def test_invalid_risk_percent_raises(self):
        with pytest.raises(ValueError):
            self.calc.calculate(100.0, direction="LONG", risk_percent=110.0)

    def test_calculate_from_exit_long(self):
        risk = self.calc.calculate_from_exit(100.0, exit_price=95.0, direction="LONG")
        assert risk.stop_loss < risk.entry_price

    def test_position_size_capped(self):
        # Tiny SL % → huge position; should be capped
        risk = self.calc.calculate(
            100.0,
            direction="LONG",
            risk_percent=2.0,
            account_size=10_000,
            sl_percent=0.01,
        )
        max_units = 10_000 * 0.20 / 100.0
        assert risk.position_size <= max_units + 0.001
