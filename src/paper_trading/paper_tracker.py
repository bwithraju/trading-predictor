"""Paper trading performance tracking and reporting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.paper_trading.paper_account import ClosedTrade, PaperAccount
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PaperPerformance:
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win_pnl: float
    avg_loss_pnl: float
    best_trade: float
    worst_trade: float
    meets_win_rate: bool
    meets_profit_factor: bool
    meets_sharpe: bool
    meets_drawdown: bool
    overall_pass: bool


class PaperTracker:
    """Track and report paper trading performance."""

    def __init__(
        self,
        account: PaperAccount,
        min_win_rate: float = 0.55,
        min_profit_factor: float = 1.5,
        min_sharpe: float = 1.0,
        max_drawdown_pct: float = 20.0,
    ):
        self.account = account
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        # Snapshot equity for drawdown calculation
        self._equity_snapshots: List[float] = [account.initial_capital]

    def record_equity(self) -> None:
        """Record current equity for drawdown tracking."""
        self._equity_snapshots.append(self.account.equity)

    def compute_performance(self) -> PaperPerformance:
        """Compute performance metrics from closed trades."""
        trades = self.account.closed_trades
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        total = len(trades)

        win_rate = len(wins) / total if total > 0 else 0.0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        best = float(max((t.pnl for t in trades), default=0.0))
        worst = float(min((t.pnl for t in trades), default=0.0))

        # Max drawdown
        snaps = np.array(self._equity_snapshots) if self._equity_snapshots else np.array(
            [self.account.initial_capital]
        )
        peak = np.maximum.accumulate(snaps)
        dd = (peak - snaps) / peak * 100
        max_dd = float(dd.max())

        # Sharpe
        pnl_pcts = [t.pnl_pct for t in trades]
        sharpe = 0.0
        if len(pnl_pcts) > 1:
            arr = np.array(pnl_pcts)
            std = arr.std()
            if std > 0:
                sharpe = float(arr.mean() / std * np.sqrt(252))

        total_pnl = self.account.equity - self.account.initial_capital
        total_pnl_pct = total_pnl / self.account.initial_capital * 100

        # Validation
        meets_wr = win_rate >= self.min_win_rate
        meets_pf = profit_factor >= self.min_profit_factor
        meets_sh = sharpe >= self.min_sharpe
        meets_dd = max_dd <= self.max_drawdown_pct

        return PaperPerformance(
            total_trades=total,
            win_trades=len(wins),
            loss_trades=len(losses),
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 4),
            avg_win_pnl=round(avg_win, 2),
            avg_loss_pnl=round(avg_loss, 2),
            best_trade=round(best, 2),
            worst_trade=round(worst, 2),
            meets_win_rate=meets_wr,
            meets_profit_factor=meets_pf,
            meets_sharpe=meets_sh,
            meets_drawdown=meets_dd,
            overall_pass=meets_wr and meets_pf and meets_sh and meets_dd,
        )

    def get_trade_journal(self) -> List[Dict]:
        """Return all closed trades as a list of dicts."""
        return [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "outcome": t.outcome,
                "exit_reason": t.exit_reason,
            }
            for t in self.account.closed_trades
        ]
