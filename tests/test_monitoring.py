"""Tests for monitoring dashboard endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_dashboard_returns_200():
    response = client.get("/monitoring/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "system_health" in data
    assert "alerts" in data


def test_health_returns_200_with_status():
    response = client.get("/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_metrics_returns_200():
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "win_rate" in data


def test_alerts_returns_200():
    response = client.get("/monitoring/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
