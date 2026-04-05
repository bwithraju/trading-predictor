"""Backtest results analysis: advanced performance metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.backtesting.engine import BacktestResult, TradeLog
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestAnalysis:
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_return_pct: float
    annual_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    # Validation against success criteria
    meets_win_rate: bool = False
    meets_profit_factor: bool = False
    meets_sharpe: bool = False
    meets_drawdown: bool = False
    meets_recovery: bool = False
    overall_pass: bool = False
    issues: List[str] = field(default_factory=list)


class BacktestAnalyzer:
    """Compute comprehensive performance metrics from backtest results."""

    def __init__(
        self,
        min_win_rate: float = 0.55,
        min_profit_factor: float = 1.5,
        min_sharpe: float = 1.0,
        max_drawdown_pct: float = 20.0,
        min_recovery_factor: float = 2.0,
    ):
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        self.min_recovery_factor = min_recovery_factor

    def analyze(
        self,
        result: BacktestResult,
        start_date: str = "2023-01-01",
        end_date: str = "2026-04-05",
        trading_days: int = 756,
    ) -> BacktestAnalysis:
        """Compute full performance analysis for a BacktestResult."""
        trades = result.trades
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]

        avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        largest_win = float(max((t.pnl_pct for t in wins), default=0.0))
        largest_loss = float(min((t.pnl_pct for t in losses), default=0.0))

        # Sortino ratio (downside deviation)
        all_returns = [t.pnl_pct for t in trades]
        neg_returns = [r for r in all_returns if r < 0]
        sortino = 0.0
        if len(all_returns) > 1 and neg_returns:
            downside_std = float(np.std(neg_returns))
            if downside_std > 0:
                sortino = float(np.mean(all_returns) / downside_std * np.sqrt(252))

        # Annualised return
        years = trading_days / 252
        annual_return = 0.0
        if years > 0:
            annual_return = (
                (1 + result.total_return_pct / 100) ** (1 / years) - 1
            ) * 100

        # Consecutive wins / losses
        max_consec_wins = self._max_consecutive(trades, "WIN")
        max_consec_losses = self._max_consecutive(trades, "LOSS")

        # Recovery factor = total return / max drawdown
        recovery = (
            abs(result.total_return_pct / result.max_drawdown_pct)
            if result.max_drawdown_pct > 0
            else float("inf")
        )

        # Validation checks
        issues = []
        meets_win_rate = result.win_rate >= self.min_win_rate
        meets_pf = result.profit_factor >= self.min_profit_factor
        meets_sharpe = result.sharpe_ratio >= self.min_sharpe
        meets_dd = result.max_drawdown_pct <= self.max_drawdown_pct
        meets_rf = recovery >= self.min_recovery_factor

        if not meets_win_rate:
            issues.append(
                f"Win rate {result.win_rate:.1%} < {self.min_win_rate:.1%}"
            )
        if not meets_pf:
            issues.append(
                f"Profit factor {result.profit_factor:.2f} < {self.min_profit_factor}"
            )
        if not meets_sharpe:
            issues.append(
                f"Sharpe ratio {result.sharpe_ratio:.2f} < {self.min_sharpe}"
            )
        if not meets_dd:
            issues.append(
                f"Max drawdown {result.max_drawdown_pct:.1f}% > {self.max_drawdown_pct}%"
            )
        if not meets_rf:
            issues.append(
                f"Recovery factor {recovery:.2f} < {self.min_recovery_factor}"
            )

        overall = meets_win_rate and meets_pf and meets_sharpe and meets_dd and meets_rf

        return BacktestAnalysis(
            symbol=result.symbol,
            start_date=start_date,
            end_date=end_date,
            total_trades=result.total_trades,
            win_rate=round(result.win_rate, 4),
            profit_factor=round(result.profit_factor, 4),
            max_drawdown_pct=round(result.max_drawdown_pct, 2),
            sharpe_ratio=round(result.sharpe_ratio, 4),
            sortino_ratio=round(sortino, 4),
            total_return_pct=round(result.total_return_pct, 2),
            annual_return_pct=round(annual_return, 2),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            largest_win_pct=round(largest_win, 4),
            largest_loss_pct=round(largest_loss, 4),
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            recovery_factor=round(recovery, 4) if recovery != float("inf") else 9999.0,
            meets_win_rate=meets_win_rate,
            meets_profit_factor=meets_pf,
            meets_sharpe=meets_sharpe,
            meets_drawdown=meets_dd,
            meets_recovery=meets_rf,
            overall_pass=overall,
            issues=issues,
        )

    @staticmethod
    def _max_consecutive(trades: List[TradeLog], outcome: str) -> int:
        """Count the maximum number of consecutive *outcome* trades."""
        max_streak = current = 0
        for t in trades:
            if t.outcome == outcome:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
