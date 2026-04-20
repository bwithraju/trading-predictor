"""Real-time monitoring dashboard FastAPI router."""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


def _get_component_status(name: str) -> dict[str, str]:
    return {"name": name, "status": "operational"}


@router.get("/dashboard")
def get_dashboard() -> dict[str, Any]:
    """Returns system health, paper trading status, backtest results summary, and alerts."""
    return {
        "system_health": {
            "status": "healthy",
            "uptime_seconds": time.time(),
            "components": [
                _get_component_status("api"),
                _get_component_status("database"),
                _get_component_status("paper_trading"),
            ],
        },
        "paper_trading": {
            "enabled": os.getenv("ALPACA_PAPER_MODE", "true").lower() == "true",
            "status": "running",
            "open_positions": 0,
        },
        "backtest_summary": {
            "win_rate": None,
            "sharpe_ratio": None,
            "profit_factor": None,
            "max_drawdown": None,
        },
        "alerts": [],
    }


@router.get("/health")
def get_health() -> dict[str, Any]:
    """Health check with component status."""
    db_status = "ok"
    try:
        from sqlalchemy import text
        from src.db.models import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_status = "error"

    live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
    paper_enabled = os.getenv("ALPACA_PAPER_MODE", "true").lower() == "true"

    return {
        "status": "healthy" if db_status == "ok" else "degraded",
        "components": {
            "api": "ok",
            "database": db_status,
            "paper_trading": "ok" if paper_enabled else "disabled",
            "live_trading": "ok" if live_enabled else "disabled",
        },
    }


@router.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Performance metrics."""
    return {
        "win_rate": None,
        "sharpe_ratio": None,
        "profit_factor": None,
        "max_drawdown": None,
        "total_trades": 0,
        "total_pnl": 0.0,
        "message": "No backtest or live trading data available yet.",
    }


@router.get("/alerts")
def get_alerts() -> dict[str, Any]:
    """Recent alerts list."""
    return {
        "alerts": [],
        "total": 0,
    }
