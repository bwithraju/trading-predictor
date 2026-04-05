"""Integration tests: test the FastAPI endpoints end-to-end."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _make_ohlcv(n=200, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    volume = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


MOCK_DF = _make_ohlcv()


class TestHealthEndpoints:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "db" in data


class TestPredictionEndpoints:
    def test_predict_stock(self):
        with patch("src.api.endpoints._load_data", return_value=MOCK_DF):
            response = client.post(
                "/predict/stock",
                json={"symbol": "AAPL", "entry_price": 150.0, "risk_percent": 2.0},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["prediction"] in ("UP", "DOWN", "NO_SIGNAL")

    def test_predict_stock_invalid_price(self):
        response = client.post(
            "/predict/stock",
            json={"symbol": "AAPL", "entry_price": -10.0},
        )
        assert response.status_code == 422

    def test_predict_stock_no_data(self):
        with patch("src.api.endpoints._load_data", return_value=pd.DataFrame()):
            response = client.post(
                "/predict/stock",
                json={"symbol": "AAPL", "entry_price": 150.0},
            )
        assert response.status_code == 404

    def test_predict_crypto(self):
        with patch("src.api.endpoints._load_data", return_value=MOCK_DF):
            response = client.post(
                "/predict/crypto",
                json={"symbol": "BTC/USDT", "entry_price": 45000.0},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("UP", "DOWN", "NO_SIGNAL")


class TestAnalysisEndpoints:
    def test_analyze_symbol(self):
        with patch("src.api.endpoints._load_data", return_value=MOCK_DF):
            response = client.get("/analyze/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "indicators" in data


class TestRiskEndpoints:
    def test_calculate_risk_basic(self):
        response = client.post(
            "/calculate-risk",
            json={"entry_price": 100.0, "direction": "LONG", "risk_percent": 2.0,
                  "account_size": 10_000.0, "reward_ratio": 2.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stop_loss"] < 100.0
        assert data["take_profit"] > 100.0

    def test_calculate_risk_invalid(self):
        response = client.post(
            "/calculate-risk",
            json={"entry_price": -1.0, "direction": "LONG"},
        )
        assert response.status_code == 422


class TestPaperTradingEndpoints:
    def test_start_paper_trading(self):
        response = client.post("/trading/paper/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "account" in data

    def test_get_paper_account(self):
        response = client.get("/trading/paper/account")
        assert response.status_code == 200
        data = response.json()
        assert "account" in data
        assert "performance" in data

    def test_execute_paper_trade_not_running(self):
        # Stop the engine first
        client.post("/trading/paper/stop")
        response = client.post(
            "/trading/paper/trade",
            json={
                "symbol": "AAPL", "direction": "LONG",
                "entry_price": 150.0, "stop_loss": 145.0,
                "take_profit": 160.0, "qty": 10.0,
            },
        )
        assert response.status_code == 400

    def test_stop_paper_trading(self):
        client.post("/trading/paper/start")
        response = client.post("/trading/paper/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert "performance" in data


class TestBacktestingEndpoints:
    def test_backtest_results_config(self):
        response = client.get("/trading/results")
        assert response.status_code == 200
        data = response.json()
        assert "default_start_date" in data
        assert "success_criteria" in data

    def test_backtest_endpoint(self):
        with patch("src.api.endpoints._load_data", return_value=MOCK_DF):
            response = client.post(
                "/backtest",
                json={"symbol": "AAPL", "asset_type": "stock",
                      "start_date": "2023-01-01", "end_date": "2026-04-05"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "report" in data


class TestTradeJournalEndpoints:
    def test_get_journal(self):
        response = client.get("/trading/journal")
        assert response.status_code == 200
        data = response.json()
        assert "open_trades" in data
        assert "closed_trades" in data

    def test_get_journal_performance(self):
        response = client.get("/trading/journal/performance")
        assert response.status_code == 200
        data = response.json()
        assert "overall" in data
        assert "by_symbol" in data

    def test_get_risk_status(self):
        response = client.get("/trading/risk-status")
        assert response.status_code == 200
        data = response.json()
        assert "can_trade" in data
        assert "daily_loss_limit_breached" in data


class TestLiveTradingEndpoints:
    def test_live_execute_disabled(self):
        # Live trading should be disabled by default
        response = client.post(
            "/trading/live/execute",
            json={
                "symbol": "AAPL", "direction": "LONG",
                "entry_price": 150.0, "stop_loss": 145.0,
                "take_profit": 160.0, "qty": 10.0,
            },
        )
        assert response.status_code == 403

    def test_positions_paper_mode(self):
        response = client.get("/trading/positions")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "paper"
