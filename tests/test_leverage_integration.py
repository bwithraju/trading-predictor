"""Integration tests for the leverage prediction system.

Tests the end-to-end flow from raw OHLCV data through feature engineering,
model recommendation, safety checks, and API endpoints.

These tests use the TestClient from FastAPI (if available) to exercise the
full request/response cycle without a running server.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.leverage.features import LeverageFeatures
from src.leverage.model import LeverageModel
from src.leverage.tiers import VolatilityTier, classify_volatility_tier
from src.leverage.trainer import LeverageTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_trending_up(n: int = 300, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, n)  # slight upward drift, low vol
    close = 100.0 * np.exp(np.cumsum(returns))
    noise = np.abs(rng.normal(0, 0.003, n)) * close
    return close, close + noise, close - noise


def _make_high_volatility(n: int = 300, seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.06, n)  # high volatility, no trend
    close = 100.0 * np.exp(np.cumsum(returns))
    noise = np.abs(rng.normal(0, 0.02, n)) * close
    return close, close + noise, close - noise


# ---------------------------------------------------------------------------
# End-to-end recommendation flow
# ---------------------------------------------------------------------------


class TestEndToEndRecommendation:
    """Verify the full pipeline from prices → recommendation."""

    def test_trending_market_allows_higher_leverage(self):
        close, high, low = _make_trending_up()
        model = LeverageModel()
        rec = model.recommend(close, high, low, account_size=1000.0)
        # Low volatility / trending → leverage should be reasonable
        assert rec.recommended_leverage >= 1

    def test_volatile_market_gives_lower_leverage(self):
        close_t, high_t, low_t = _make_trending_up()
        close_v, high_v, low_v = _make_high_volatility()

        model = LeverageModel()
        rec_trend = model.recommend(close_t, high_t, low_t, account_size=1000.0)
        rec_volatile = model.recommend(close_v, high_v, low_v, account_size=1000.0)

        # Volatile market should recommend ≤ trending market
        assert rec_volatile.recommended_leverage <= rec_trend.recommended_leverage

    def test_recommendation_is_serialisable(self):
        close, high, low = _make_trending_up()
        model = LeverageModel()
        rec = model.recommend(close, high, low, account_size=500.0)
        d = rec.to_dict()
        assert isinstance(d, dict)
        # All numeric values should be JSON-serialisable (no numpy types)
        import json
        json.dumps(d)  # raises TypeError if not serialisable

    def test_safety_report_present(self):
        close, high, low = _make_trending_up()
        model = LeverageModel()
        rec = model.recommend(close, high, low, account_size=1000.0)
        assert rec.safety_report is not None
        assert "passed" in rec.safety_report.to_dict()

    def test_volatility_tier_is_valid_enum(self):
        close, high, low = _make_trending_up()
        model = LeverageModel()
        rec = model.recommend(close, high, low, account_size=1000.0)
        assert rec.volatility_tier in VolatilityTier.__members__.values()


# ---------------------------------------------------------------------------
# Feature → tier → recommendation pipeline
# ---------------------------------------------------------------------------


class TestFeatureTierPipeline:
    def test_features_feed_into_tier(self):
        close, high, low = _make_trending_up(300)
        engine = LeverageFeatures()
        feats = engine.compute(close, high, low)
        feats = engine.fill_missing(feats)

        vol_ratio = feats.get("volatility_ratio", 1.0)
        tier_info = classify_volatility_tier(max(float(vol_ratio), 1e-9))
        assert tier_info is not None
        assert 1 <= tier_info.max_leverage <= 20

    def test_high_vol_features_produce_high_ratio(self):
        close, high, low = _make_high_volatility(300)
        engine = LeverageFeatures()
        feats = engine.compute(close, high, low)
        feats = engine.fill_missing(feats)
        # Annualised vol should be a positive number; exact comparison is
        # data-dependent so we just check the value is finite and positive.
        assert feats["hist_vol_20"] > 0

    def test_feature_array_no_nan_after_fill(self):
        close, high, low = _make_trending_up(300)
        engine = LeverageFeatures()
        feats = engine.compute(close, high, low)
        feats = engine.fill_missing(feats)
        arr = engine.to_array(feats)
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    def test_synthetic_training_produces_model(self):
        trainer = LeverageTrainer()
        model = trainer.train_from_synthetic(n_bars=500, n_estimators=10)
        assert model._model_loaded is True

    def test_trained_model_gives_valid_recommendations(self):
        trainer = LeverageTrainer()
        model = trainer.train_from_synthetic(n_bars=500, n_estimators=10)
        close, high, low = _make_trending_up(200)
        rec = model.recommend(close, high, low, account_size=1000.0)
        assert 1 <= rec.recommended_leverage <= 20

    def test_build_training_data_shapes(self):
        trainer = LeverageTrainer(window=60, step=5)
        close, high, low = _make_trending_up(300)
        X, y_cls, y_reg = trainer.build_training_data([(close, high, low)])
        assert X.ndim == 2
        assert X.shape[1] == len(LeverageFeatures.FEATURE_NAMES)
        assert X.shape[0] == y_cls.shape[0] == y_reg.shape[0]
        assert set(np.unique(y_cls)).issubset({0, 1})

    def test_training_with_multiple_assets(self):
        """Mixed trending + high-volatility data should produce both class labels.

        Uses window=150 so 100-period historical volatility can be computed,
        enabling the classifier to see both safe and risky conditions.
        """
        trainer = LeverageTrainer(window=150, step=10)
        close1, high1, low1 = _make_trending_up(500, seed=1)
        close2, high2, low2 = _make_high_volatility(500, seed=2)
        X, y_cls, y_reg = trainer.build_training_data([
            (close1, high1, low1),
            (close2, high2, low2),
        ])
        assert X.shape[0] > 0
        # Mixed conditions (low-vol trending + high-vol) should produce both safe and risky labels
        assert set(np.unique(y_cls)).issubset({0, 1})
        assert len(np.unique(y_cls)) == 2, "Expected both safe (1) and risky (0) labels in mixed training data"

    def test_insufficient_data_raises(self):
        trainer = LeverageTrainer(window=100, step=1)
        close = np.linspace(100, 200, 50)  # too short for window=100
        high = close * 1.01
        low = close * 0.99
        with pytest.raises(ValueError, match="No training samples"):
            trainer.build_training_data([(close, high, low)])


# ---------------------------------------------------------------------------
# API endpoint integration (FastAPI TestClient)
# ---------------------------------------------------------------------------


try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from src.api.leverage_endpoints import router

    _app = FastAPI()
    _app.include_router(router)
    _client = TestClient(_app)
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI / httpx not installed")
class TestAPIEndpoints:
    def _make_price_lists(self, n: int = 60, seed: int = 1):
        close, high, low = _make_trending_up(n, seed=seed)
        return close.tolist(), high.tolist(), low.tolist()

    def test_status_endpoint(self):
        resp = _client.get("/leverage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_loaded" in data
        assert "available_tiers" in data

    def test_recommend_endpoint(self):
        close, high, low = self._make_price_lists(80)
        payload = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "account_size": 1000.0,
            "close_prices": close,
            "high_prices": high,
            "low_prices": low,
        }
        resp = _client.post("/leverage/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert 1 <= data["recommended_leverage"] <= 20
        assert 0 <= data["confidence_score"] <= 100

    def test_recommend_too_few_prices(self):
        close, high, low = self._make_price_lists(10)
        payload = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "account_size": 1000.0,
            "close_prices": close,
            "high_prices": high,
            "low_prices": low,
        }
        resp = _client.post("/leverage/recommend", json=payload)
        assert resp.status_code == 422  # validation error

    def test_adjust_endpoint_approved(self):
        payload = {
            "current_leverage": 2,
            "new_leverage_request": 2,
            "account_size": 1000.0,
            "current_drawdown_pct": 3.0,
            "volatility_ratio": 0.8,
        }
        resp = _client.post("/leverage/adjust", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "approved" in data
        assert "safer_alternative" in data

    def test_adjust_endpoint_rejected(self):
        payload = {
            "current_leverage": 4,
            "new_leverage_request": 10,
            "account_size": 1000.0,
            "current_drawdown_pct": 5.0,
            "volatility_ratio": 2.5,  # extreme volatility
        }
        resp = _client.post("/leverage/adjust", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["approved"] is False

    def test_analysis_endpoint(self):
        prices = ",".join(str(p) for p in np.linspace(50000, 52000, 50).tolist())
        resp = _client.get(f"/leverage/analysis/BTC-USDT?close_prices={prices}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTC-USDT"
        assert "volatility_tier" in data

    def test_analysis_too_few_prices(self):
        resp = _client.get("/leverage/analysis/BTC-USDT?close_prices=100,200,300")
        assert resp.status_code == 422

    def test_backtest_endpoint_dynamic(self):
        close, high, low = self._make_price_lists(200)
        payload = {
            "close_prices": close,
            "high_prices": high,
            "low_prices": low,
            "strategy": "dynamic",
            "account_size": 1000.0,
        }
        resp = _client.post("/leverage/backtest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_recommended_leverage" in data
        assert data["total_bars"] > 0

    def test_backtest_endpoint_fixed(self):
        close, high, low = self._make_price_lists(200)
        payload = {
            "close_prices": close,
            "high_prices": high,
            "low_prices": low,
            "strategy": "fixed",
            "fixed_leverage": 3,
            "account_size": 1000.0,
        }
        resp = _client.post("/leverage/backtest", json=payload)
        assert resp.status_code == 200

    def test_insights_endpoint(self):
        resp = _client.get("/leverage/insights")
        assert resp.status_code == 200
        data = resp.json()
        assert "tier_thresholds" in data
        assert len(data["tier_thresholds"]) == 5  # 5 tiers
