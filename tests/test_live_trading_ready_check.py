"""Tests for live trading readiness checker."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from live_trading.ready_check import LiveTradingReadyCheck


@pytest.fixture
def checker() -> LiveTradingReadyCheck:
    return LiveTradingReadyCheck()


GOOD_METRICS = {
    "win_rate": 0.60,
    "profit_factor": 2.0,
    "sharpe_ratio": 1.5,
    "max_drawdown": 0.10,
}

BAD_METRICS = {
    "win_rate": 0.40,
    "profit_factor": 0.8,
    "sharpe_ratio": 0.5,
    "max_drawdown": 0.35,
}


def test_instantiation(checker):
    assert isinstance(checker, LiveTradingReadyCheck)


def test_check_backtest_metrics_passes_with_good_metrics(checker):
    result = checker.check_backtest_metrics(GOOD_METRICS)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_check_backtest_metrics_fails_with_bad_metrics(checker):
    result = checker.check_backtest_metrics(BAD_METRICS)
    assert result["passed"] is False


def test_run_all_checks_returns_dict(checker):
    start = date.today() - timedelta(days=10)
    results = checker.run_all_checks(
        backtest_results=GOOD_METRICS,
        paper_results=GOOD_METRICS,
        paper_start_date=start,
    )
    assert isinstance(results, dict)
    assert "backtest_metrics" in results
    assert "paper_trading_duration" in results
    assert "risk_management" in results
    assert "monitoring_systems" in results
    assert "overall_passed" in results


def test_generate_checklist_report_returns_markdown(checker):
    checker.run_all_checks(backtest_results=GOOD_METRICS)
    report = checker.generate_checklist_report()
    assert isinstance(report, str)
    assert "# Live Trading Readiness Checklist" in report
    assert "Overall" in report


def test_paper_trading_duration_passes_when_old_enough(checker):
    start = date.today() - timedelta(days=10)
    result = checker.check_paper_trading_duration(start)
    assert result["passed"] is True


def test_paper_trading_duration_fails_when_too_recent(checker):
    start = date.today() - timedelta(days=3)
    result = checker.check_paper_trading_duration(start)
    assert result["passed"] is False
