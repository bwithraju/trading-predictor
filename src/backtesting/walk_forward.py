"""Walk-forward out-of-sample validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    symbol: str
    n_windows: int
    window_results: List[BacktestResult] = field(default_factory=list)
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    avg_sharpe_ratio: float = 0.0
    avg_max_drawdown: float = 0.0
    avg_total_return: float = 0.0
    is_robust: bool = False


class WalkForwardAnalyzer:
    """Validate strategy robustness via rolling walk-forward windows."""

    def __init__(
        self,
        train_pct: float = 0.7,
        n_windows: int = 4,
    ):
        self.train_pct = train_pct
        self.n_windows = n_windows

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        risk_percent: float = 2.0,
        account_size: float = 10_000.0,
    ) -> WalkForwardResult:
        """Run walk-forward analysis by splitting *df* into N sliding windows."""
        total_bars = len(df)
        if total_bars < 100:
            logger.warning("Insufficient data for walk-forward analysis (%d bars)", total_bars)
            return WalkForwardResult(symbol=symbol, n_windows=0)

        window_size = total_bars // self.n_windows
        results: List[BacktestResult] = []

        for i in range(self.n_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, total_bars)
            window_df = df.iloc[start_idx:end_idx].copy()

            if len(window_df) < 60:
                logger.warning("Skipping window %d (too few rows: %d)", i, len(window_df))
                continue

            logger.info(
                "Walk-forward window %d/%d: bars %d–%d",
                i + 1, self.n_windows, start_idx, end_idx,
            )

            engine = BacktestEngine()
            try:
                res = engine.run(
                    df=window_df,
                    symbol=symbol,
                    risk_percent=risk_percent,
                    account_size=account_size,
                )
                results.append(res)
            except Exception as exc:
                logger.error("Walk-forward window %d failed: %s", i, exc)

        if not results:
            return WalkForwardResult(symbol=symbol, n_windows=0)

        avg_win_rate = sum(r.win_rate for r in results) / len(results)
        avg_pf = sum(r.profit_factor for r in results) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_dd = sum(r.max_drawdown_pct for r in results) / len(results)
        avg_ret = sum(r.total_return_pct for r in results) / len(results)

        # Robust if avg win rate > 50% and profit factor > 1.0
        is_robust = avg_win_rate > 0.50 and avg_pf > 1.0

        return WalkForwardResult(
            symbol=symbol,
            n_windows=len(results),
            window_results=results,
            avg_win_rate=round(avg_win_rate, 4),
            avg_profit_factor=round(avg_pf, 4),
            avg_sharpe_ratio=round(avg_sharpe, 4),
            avg_max_drawdown=round(avg_dd, 2),
            avg_total_return=round(avg_ret, 2),
            is_robust=is_robust,
        )
