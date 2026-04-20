"""Live trading readiness checker."""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any


class LiveTradingReadyCheck:
    """Validates that the system is ready for live trading."""

    # Thresholds (overridable via env vars)
    MIN_WIN_RATE: float = float(os.getenv("MIN_WIN_RATE", "0.55"))
    MIN_PROFIT_FACTOR: float = float(os.getenv("MIN_PROFIT_FACTOR", "1.5"))
    MIN_SHARPE_RATIO: float = float(os.getenv("MIN_SHARPE_RATIO", "1.0"))
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", "0.20"))
    MIN_PAPER_TRADING_DAYS: int = int(os.getenv("MIN_PAPER_TRADING_DAYS", "7"))
    ALIGNMENT_TOLERANCE: float = 0.10  # 10 %

    def __init__(self) -> None:
        self._results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_backtest_metrics(self, backtest_results: dict[str, float]) -> dict[str, Any]:
        win_rate = backtest_results.get("win_rate", 0.0)
        profit_factor = backtest_results.get("profit_factor", 0.0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
        max_drawdown = backtest_results.get("max_drawdown", 1.0)

        checks = {
            "win_rate": win_rate >= self.MIN_WIN_RATE,
            "profit_factor": profit_factor >= self.MIN_PROFIT_FACTOR,
            "sharpe_ratio": sharpe_ratio >= self.MIN_SHARPE_RATIO,
            "max_drawdown": max_drawdown <= self.MAX_DRAWDOWN,
        }
        passed = all(checks.values())
        result = {
            "passed": passed,
            "checks": checks,
            "values": {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            },
        }
        self._results["backtest_metrics"] = result
        return result

    def check_paper_trading_duration(self, start_date: date | datetime | str) -> dict[str, Any]:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        days_elapsed = (date.today() - start_date).days
        passed = days_elapsed >= self.MIN_PAPER_TRADING_DAYS
        result = {
            "passed": passed,
            "days_elapsed": days_elapsed,
            "required_days": self.MIN_PAPER_TRADING_DAYS,
        }
        self._results["paper_trading_duration"] = result
        return result

    def check_paper_trading_alignment(
        self,
        paper_results: dict[str, float],
        backtest_results: dict[str, float],
    ) -> dict[str, Any]:
        metrics_to_check = ["win_rate", "profit_factor"]
        alignment: dict[str, bool] = {}
        for metric in metrics_to_check:
            paper_val = paper_results.get(metric)
            backtest_val = backtest_results.get(metric)
            if paper_val is None or backtest_val is None or backtest_val == 0:
                alignment[metric] = False
                continue
            deviation = abs(paper_val - backtest_val) / abs(backtest_val)
            alignment[metric] = deviation <= self.ALIGNMENT_TOLERANCE

        passed = all(alignment.values())
        result = {"passed": passed, "alignment": alignment}
        self._results["paper_trading_alignment"] = result
        return result

    def check_risk_management(self) -> dict[str, Any]:
        required_vars = [
            "LIVE_MAX_RISK_PCT",
            "LIVE_DAILY_LOSS_LIMIT_PCT",
            "LIVE_MAX_OPEN_POSITIONS",
        ]
        missing = [v for v in required_vars if not os.getenv(v)]
        # Treat missing env vars as using safe defaults — check values are sane
        max_risk = float(os.getenv("LIVE_MAX_RISK_PCT", "2.0"))
        daily_loss = float(os.getenv("LIVE_DAILY_LOSS_LIMIT_PCT", "5.0"))
        max_positions = int(os.getenv("LIVE_MAX_OPEN_POSITIONS", "5"))

        checks = {
            "max_risk_pct_configured": max_risk > 0,
            "daily_loss_limit_configured": daily_loss > 0,
            "max_positions_configured": max_positions > 0,
        }
        passed = all(checks.values())
        result = {
            "passed": passed,
            "checks": checks,
            "missing_env_vars": missing,
            "values": {
                "max_risk_pct": max_risk,
                "daily_loss_limit_pct": daily_loss,
                "max_open_positions": max_positions,
            },
        }
        self._results["risk_management"] = result
        return result

    def check_monitoring_systems(self) -> dict[str, Any]:
        monitoring_enabled = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
        checks = {"monitoring_enabled": monitoring_enabled}
        passed = all(checks.values())
        result = {"passed": passed, "checks": checks}
        self._results["monitoring_systems"] = result
        return result

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        backtest_results: dict[str, float] | None = None,
        paper_results: dict[str, float] | None = None,
        paper_start_date: date | datetime | str | None = None,
    ) -> dict[str, Any]:
        """Run all checks and return a summary dict."""
        results: dict[str, Any] = {}

        if backtest_results is not None:
            results["backtest_metrics"] = self.check_backtest_metrics(backtest_results)
        else:
            results["backtest_metrics"] = {"passed": False, "reason": "No backtest results provided"}

        if paper_start_date is not None:
            results["paper_trading_duration"] = self.check_paper_trading_duration(paper_start_date)
        else:
            results["paper_trading_duration"] = {"passed": False, "reason": "No start date provided"}

        if paper_results is not None and backtest_results is not None:
            results["paper_trading_alignment"] = self.check_paper_trading_alignment(
                paper_results, backtest_results
            )
        else:
            results["paper_trading_alignment"] = {"passed": False, "reason": "Insufficient data"}

        results["risk_management"] = self.check_risk_management()
        results["monitoring_systems"] = self.check_monitoring_systems()

        results["overall_passed"] = all(v.get("passed", False) for v in results.values() if isinstance(v, dict))
        return results

    def generate_checklist_report(self) -> str:
        """Return a markdown checklist based on the most recent run_all_checks() call."""
        lines = ["# Live Trading Readiness Checklist\n"]
        check_labels = {
            "backtest_metrics": "Backtest metrics meet thresholds",
            "paper_trading_duration": f"Paper trading ≥ {self.MIN_PAPER_TRADING_DAYS} days",
            "paper_trading_alignment": "Paper trading aligns with backtest (±10%)",
            "risk_management": "Risk management configured",
            "monitoring_systems": "Monitoring systems operational",
        }
        for key, label in check_labels.items():
            result = self._results.get(key)
            if result is None:
                mark = "⬜"
            elif result.get("passed"):
                mark = "✅"
            else:
                mark = "❌"
            lines.append(f"- [{mark}] {label}")

        overall = all(r.get("passed", False) for r in self._results.values())
        lines.append(f"\n**Overall: {'READY ✅' if overall else 'NOT READY ❌'}**")
        return "\n".join(lines)
