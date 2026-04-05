"""Tests for the alert notification service."""
from __future__ import annotations

import pytest

from alerts.notification_service import AlertType, NotificationService


@pytest.fixture
def svc() -> NotificationService:
    return NotificationService()


def test_instantiation(svc):
    assert isinstance(svc, NotificationService)


def test_send_alert_creates_entry(svc):
    entry = svc.send_alert(AlertType.SYSTEM_ERROR, "test error")
    assert entry["type"] == AlertType.SYSTEM_ERROR.value
    assert entry["message"] == "test error"
    assert "timestamp" in entry


def test_check_daily_loss_triggers_alert_when_exceeded(svc):
    # Force threshold to a known value
    svc.DAILY_LOSS_THRESHOLD = 0.05
    fired = svc.check_daily_loss(0.06)
    assert fired is True
    history = svc.get_alert_history()
    assert any(a["type"] == AlertType.DAILY_LOSS.value for a in history)


def test_check_daily_loss_no_alert_below_threshold(svc):
    svc.DAILY_LOSS_THRESHOLD = 0.05
    fired = svc.check_daily_loss(0.03)
    assert fired is False


def test_check_performance_milestone_triggers(svc):
    metrics = {"win_rate": 0.65, "sharpe_ratio": 2.0}
    fired = svc.check_performance_milestone(metrics)
    assert fired is True
    history = svc.get_alert_history()
    assert any(a["type"] == AlertType.PERFORMANCE_MILESTONE.value for a in history)


def test_check_performance_milestone_no_trigger(svc):
    metrics = {"win_rate": 0.50, "sharpe_ratio": 0.8}
    fired = svc.check_performance_milestone(metrics)
    assert fired is False


def test_get_alert_history_returns_list(svc):
    svc.send_alert(AlertType.UNUSUAL_ACTIVITY, "unusual")
    history = svc.get_alert_history()
    assert isinstance(history, list)
    assert len(history) >= 1
