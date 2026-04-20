"""Trade analysis: performance breakdown from the trade journal."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from src.trade_management.trade_logger import TradeLogger
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeAnalyzer:
    """Analyse trades from the journal for performance reporting."""

    def __init__(self, trade_logger: TradeLogger):
        self.trade_logger = trade_logger

    def analyze_all(self) -> Dict[str, Any]:
        """Return aggregate performance across all closed trades."""
        trades = self.trade_logger.load_closed_trades()
        return self._compute_stats(trades, label="all")

    def analyze_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Return per-symbol performance breakdown."""
        trades = self.trade_logger.load_closed_trades()
        by_symbol: Dict[str, List[dict]] = defaultdict(list)
        for t in trades:
            by_symbol[t["symbol"]].append(t)
        return {sym: self._compute_stats(ts, label=sym) for sym, ts in by_symbol.items()}

    def analyze_by_mode(self) -> Dict[str, Dict[str, Any]]:
        """Return per-mode (paper/live) performance breakdown."""
        trades = self.trade_logger.load_closed_trades()
        by_mode: Dict[str, List[dict]] = defaultdict(list)
        for t in trades:
            by_mode[t.get("mode", "paper")].append(t)
        return {mode: self._compute_stats(ts, label=mode) for mode, ts in by_mode.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(trades: List[dict], label: str) -> Dict[str, Any]:
        total = len(trades)
        if total == 0:
            return {
                "label": label,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        wins = [t for t in trades if t.get("outcome") == "WIN"]
        losses = [t for t in trades if t.get("outcome") == "LOSS"]
        win_rate = len(wins) / total

        gross_profit = sum(t.get("pnl", 0.0) for t in wins)
        gross_loss = abs(sum(t.get("pnl", 0.0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        pnl_values = [t.get("pnl", 0.0) for t in trades]
        total_pnl = sum(pnl_values)
        avg_pnl = total_pnl / total

        pnl_arr = np.array(pnl_values)
        sharpe = 0.0
        if len(pnl_arr) > 1 and pnl_arr.std() > 0:
            sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))

        return {
            "label": label,
            "total_trades": total,
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": round(max(pnl_values), 2) if pnl_values else 0.0,
            "worst_trade": round(min(pnl_values), 2) if pnl_values else 0.0,
            "sharpe_ratio": round(sharpe, 4),
        }
