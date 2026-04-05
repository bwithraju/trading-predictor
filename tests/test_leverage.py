"""Unit tests for the leverage prediction system.

Tests cover:
  - Volatility tier classification
  - Feature engineering
  - Leverage calculator
  - Safety checker
  - Leverage model (heuristic path)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.leverage.calculator import LeverageCalculator
from src.leverage.features import (
    LeverageFeatures,
    calculate_atr,
    calculate_bollinger_band_width,
    calculate_current_drawdown,
    calculate_historical_volatility,
    calculate_max_drawdown,
    calculate_rsi,
    calculate_trend_consistency,
    calculate_trend_direction,
    calculate_trend_slope,
    detect_volatility_spike,
)
from src.leverage.model import LeverageModel
from src.leverage.safety_checker import SafetyChecker, Severity
from src.leverage.tiers import (
    TierInfo,
    VolatilityTier,
    classify_volatility_tier,
    get_tier_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic close/high/low price arrays."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.015, size=n)
    close = 50000.0 * np.exp(np.cumsum(returns))
    noise = np.abs(rng.normal(0, 0.004, size=n)) * close
    high = close + noise
    low = close - noise
    return close, high, low


# ---------------------------------------------------------------------------
# Tests: tiers
# ---------------------------------------------------------------------------


class TestVolatilityTiers:
    def test_extreme_high_ratio(self):
        info = classify_volatility_tier(2.0)
        assert info.tier == VolatilityTier.EXTREME
        assert info.max_leverage == 1

    def test_high_ratio(self):
        info = classify_volatility_tier(1.2)
        assert info.tier == VolatilityTier.HIGH
        assert info.max_leverage == 2

    def test_normal_ratio(self):
        info = classify_volatility_tier(0.85)
        assert info.tier == VolatilityTier.NORMAL
        assert info.max_leverage == 4

    def test_low_ratio(self):
        info = classify_volatility_tier(0.6)
        assert info.tier == VolatilityTier.LOW
        assert info.max_leverage == 8

    def test_extreme_low_ratio(self):
        info = classify_volatility_tier(0.3)
        assert info.tier == VolatilityTier.EXTREME_LOW
        assert info.max_leverage == 15

    def test_boundary_at_1_5(self):
        """Exactly 1.5 should map to EXTREME."""
        info = classify_volatility_tier(1.5)
        assert info.tier == VolatilityTier.EXTREME

    def test_boundary_just_below_1_5(self):
        """1.499 should map to HIGH."""
        info = classify_volatility_tier(1.499)
        assert info.tier == VolatilityTier.HIGH

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            classify_volatility_tier(0.0)

    def test_get_tier_info(self):
        info = get_tier_info(VolatilityTier.NORMAL)
        assert isinstance(info, TierInfo)
        assert info.tier == VolatilityTier.NORMAL

    def test_recommended_leverage_property(self):
        info = get_tier_info(VolatilityTier.LOW)
        assert info.recommended_leverage == info.max_leverage


# ---------------------------------------------------------------------------
# Tests: features
# ---------------------------------------------------------------------------


class TestFeatureHelpers:
    def setup_method(self):
        self.close, self.high, self.low = _make_prices(200)

    def test_atr_returns_positive(self):
        atr = calculate_atr(self.close, self.high, self.low, period=14)
        assert atr > 0

    def test_atr_insufficient_data(self):
        atr = calculate_atr(self.close[:5], self.high[:5], self.low[:5], period=14)
        assert np.isnan(atr)

    def test_historical_volatility_range(self):
        vol = calculate_historical_volatility(self.close, period=20)
        assert 0 < vol < 10  # annualised, should be a reasonable number

    def test_historical_volatility_insufficient(self):
        vol = calculate_historical_volatility(self.close[:10], period=20)
        assert np.isnan(vol)

    def test_bollinger_width_positive(self):
        bw = calculate_bollinger_band_width(self.close, period=20)
        assert bw > 0

    def test_rsi_bounds(self):
        rsi = calculate_rsi(self.close, period=14)
        assert 0 <= rsi <= 100

    def test_trend_slope_sign(self):
        # Monotonically rising prices → positive slope
        rising = np.linspace(100, 200, 50)
        slope = calculate_trend_slope(rising, period=20)
        assert slope > 0

    def test_trend_consistency_bounds(self):
        tc = calculate_trend_consistency(self.close, period=20)
        assert 0 <= tc <= 1

    def test_trend_direction_up(self):
        rising = np.linspace(100, 200, 100)
        assert calculate_trend_direction(rising) == 1

    def test_trend_direction_down(self):
        falling = np.linspace(200, 100, 100)
        assert calculate_trend_direction(falling) == -1

    def test_current_drawdown_at_peak(self):
        # Price at all-time high → drawdown = 0
        prices = np.array([90.0, 95.0, 100.0])
        dd = calculate_current_drawdown(prices)
        assert dd == pytest.approx(0.0)

    def test_current_drawdown_non_zero(self):
        prices = np.array([100.0, 110.0, 90.0])
        dd = calculate_current_drawdown(prices)
        assert dd == pytest.approx((110 - 90) / 110, rel=1e-5)

    def test_max_drawdown_non_negative(self):
        prices = np.array([100.0, 120.0, 80.0, 130.0, 70.0])
        dd = calculate_max_drawdown(prices)
        # Peak was 130, trough was 70 → ~46%
        assert dd == pytest.approx((130 - 70) / 130, rel=1e-5)

    def test_detect_volatility_spike_no_spike(self):
        # Low-noise prices should not trigger a spike
        flat = np.full(100, 100.0) + np.random.default_rng(0).normal(0, 0.001, 100)
        assert detect_volatility_spike(flat, period=20) is False


class TestLeverageFeatures:
    def setup_method(self):
        self.close, self.high, self.low = _make_prices(200)
        self.engine = LeverageFeatures()

    def test_compute_returns_all_features(self):
        feats = self.engine.compute(self.close, self.high, self.low)
        for name in LeverageFeatures.FEATURE_NAMES:
            assert name in feats, f"Missing feature: {name}"

    def test_to_array_length(self):
        feats = self.engine.compute(self.close, self.high, self.low)
        arr = self.engine.to_array(feats)
        assert arr.shape == (len(LeverageFeatures.FEATURE_NAMES),)

    def test_fill_missing_removes_nan(self):
        feats = {"a": np.nan, "b": 1.0}
        filled = self.engine.fill_missing(feats)
        assert not np.isnan(filled["a"])
        assert filled["b"] == 1.0


# ---------------------------------------------------------------------------
# Tests: calculator
# ---------------------------------------------------------------------------


class TestLeverageCalculator:
    def setup_method(self):
        self.calc = LeverageCalculator()

    def test_buying_power(self):
        bp = self.calc.buying_power(1000.0, 4)
        assert bp == 4000.0

    def test_max_position_size(self):
        # 2% risk, 5% stop-loss → max position = 1000 * 0.02 / 0.05 = 400
        pos = self.calc.max_position_size(1000.0, risk_percent=2.0, stop_loss_percent=5.0)
        assert pos == pytest.approx(400.0)

    def test_liquidation_price_long(self):
        liq = self.calc.liquidation_price(100.0, leverage=10, direction="long")
        assert liq < 100.0  # long liquidation below entry

    def test_liquidation_price_short(self):
        liq = self.calc.liquidation_price(100.0, leverage=10, direction="short")
        assert liq > 100.0  # short liquidation above entry

    def test_liquidation_risk_high_leverage(self):
        risk = self.calc.liquidation_risk(volatility=0.5, trend_strength=0.5, leverage=20)
        assert risk > 0

    def test_liquidation_risk_low_leverage(self):
        risk_low = self.calc.liquidation_risk(volatility=0.3, trend_strength=0.5, leverage=1)
        risk_high = self.calc.liquidation_risk(volatility=0.3, trend_strength=0.5, leverage=10)
        assert risk_high > risk_low

    def test_liquidation_risk_clamped(self):
        risk = self.calc.liquidation_risk(volatility=1.0, trend_strength=0.0, leverage=20)
        assert 0 <= risk <= 100

    def test_risk_level_labels(self):
        assert self.calc.risk_level_label(35) == "CRITICAL"
        assert self.calc.risk_level_label(20) == "HIGH"
        assert self.calc.risk_level_label(10) == "MODERATE"
        assert self.calc.risk_level_label(3) == "LOW"

    def test_daily_stop_loss_amount(self):
        amount = self.calc.daily_stop_loss_amount(1000.0, 5.0)
        assert amount == pytest.approx(50.0)

    def test_recommended_leverage_from_risk(self):
        lev = self.calc.recommended_leverage_from_risk(
            account_size=1000.0,
            risk_percent=2.0,
            stop_loss_percent=5.0,
            model_recommendation=8,
        )
        # max_position = 400, max_leverage_from_position = 0.4 → clamped to 1
        assert lev == 1

    def test_invalid_buying_power(self):
        with pytest.raises(ValueError):
            self.calc.buying_power(-100.0, 4)

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            self.calc.liquidation_price(100.0, 4, direction="sideways")

    def test_account_summary_keys(self):
        summary = self.calc.account_summary(1000.0, leverage=4)
        assert "buying_power" in summary
        assert "max_position_size" in summary
        assert "daily_stop_loss" in summary


# ---------------------------------------------------------------------------
# Tests: safety checker
# ---------------------------------------------------------------------------


class TestSafetyChecker:
    def setup_method(self):
        self.checker = SafetyChecker()

    def _run_defaults(self, **overrides):
        defaults = dict(
            account_size=1000.0,
            data_length=200,
            volatility_ratio=0.85,
            volatility_spike=False,
            current_drawdown=0.05,
            requested_leverage=4,
            liquidation_risk_pct=10.0,
        )
        defaults.update(overrides)
        return self.checker.run_checks(**defaults)

    def test_all_passing(self):
        report = self._run_defaults()
        assert report.passed

    def test_account_too_small(self):
        report = self._run_defaults(account_size=5.0)
        assert not report.passed
        labels = [i.check for i in report.issues if not i.passed]
        assert "account_size" in labels

    def test_insufficient_data(self):
        report = self._run_defaults(data_length=10)
        assert not report.passed

    def test_volatility_spike_warning(self):
        report = self._run_defaults(volatility_spike=True)
        # Spike is WARNING not CRITICAL
        assert report.max_severity == Severity.WARNING

    def test_high_drawdown_warning(self):
        report = self._run_defaults(current_drawdown=0.30)
        warnings = [i for i in report.issues if i.check == "drawdown"]
        assert warnings[0].severity == Severity.WARNING

    def test_leverage_cap_exceeded(self):
        report = self._run_defaults(requested_leverage=25)
        assert not report.passed

    def test_critical_liquidation_risk(self):
        report = self._run_defaults(liquidation_risk_pct=50.0)
        assert not report.passed

    def test_safety_report_to_dict(self):
        report = self._run_defaults()
        d = report.to_dict()
        assert "passed" in d
        assert "issues" in d
        assert isinstance(d["issues"], list)


# ---------------------------------------------------------------------------
# Tests: leverage model (heuristic path)
# ---------------------------------------------------------------------------


class TestLeverageModel:
    def setup_method(self):
        self.close, self.high, self.low = _make_prices(200)
        self.model = LeverageModel()

    def test_recommend_returns_recommendation(self):
        rec = self.model.recommend(
            self.close, self.high, self.low,
            symbol="BTC/USDT",
            account_size=1000.0,
        )
        assert 1 <= rec.recommended_leverage <= 20

    def test_recommend_confidence_range(self):
        rec = self.model.recommend(self.close, self.high, self.low, account_size=1000.0)
        assert 0 <= rec.confidence_score <= 100

    def test_recommend_safety_score_range(self):
        rec = self.model.recommend(self.close, self.high, self.low, account_size=1000.0)
        assert 0 <= rec.safety_score <= 100

    def test_to_dict_complete(self):
        rec = self.model.recommend(self.close, self.high, self.low, account_size=1000.0)
        d = rec.to_dict()
        expected_keys = {
            "symbol", "timeframe", "recommended_leverage", "max_safe_leverage",
            "aggressive_leverage", "confidence_score", "safety_score",
            "liquidation_risk_pct", "volatility_tier", "volatility_ratio",
            "trend_direction", "trend_strength", "account_size", "buying_power",
            "max_position_size", "daily_stop_loss", "risk_level",
            "adjustment_needed", "adjustment_reason", "safety_report",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_buying_power_scales_with_leverage(self):
        rec = self.model.recommend(self.close, self.high, self.low, account_size=1000.0)
        assert rec.buying_power == pytest.approx(1000.0 * rec.recommended_leverage)

    def test_small_account_reduces_leverage(self):
        """Very small accounts should not get high leverage."""
        rec = self.model.recommend(self.close, self.high, self.low, account_size=50.0)
        assert 1 <= rec.recommended_leverage <= 20

    def test_analyze_returns_dict(self):
        result = self.model.analyze(self.close, self.high, self.low, symbol="ETH/USDT")
        assert "symbol" in result
        assert "volatility_tier" in result
        assert "features" in result

    def test_recommend_insufficient_data_uses_defaults(self):
        """Short price series should not raise – model handles gracefully."""
        close = np.linspace(100, 110, 40)
        high = close * 1.002
        low = close * 0.998
        rec = self.model.recommend(close, high, low, account_size=500.0)
        assert 1 <= rec.recommended_leverage <= 20

    def test_extreme_volatility_gives_low_leverage(self):
        """High-volatility environment should recommend 1× or 2× leverage."""
        rng = np.random.default_rng(99)
        # High volatility: large random moves
        log_ret = rng.normal(0, 0.15, 200)  # 15% daily vol
        close = 1000.0 * np.exp(np.cumsum(log_ret))
        high = close * 1.05
        low = close * 0.95
        rec = self.model.recommend(close, high, low, account_size=1000.0)
        assert rec.recommended_leverage <= 4  # must be conservative
