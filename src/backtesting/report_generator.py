"""Generate text-based backtest reports."""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from src.backtesting.backtest_analyzer import BacktestAnalysis
from src.backtesting.walk_forward import WalkForwardResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DIVIDER = "=" * 60


class ReportGenerator:
    """Produce human-readable performance reports."""

    def generate_backtest_report(
        self,
        analysis: BacktestAnalysis,
        walk_forward: Optional[WalkForwardResult] = None,
    ) -> str:
        """Return a formatted text report for a backtest analysis."""
        lines = [
            _DIVIDER,
            f"  BACKTEST REPORT: {analysis.symbol}",
            f"  Period: {analysis.start_date} → {analysis.end_date}",
            _DIVIDER,
            "",
            "── Performance Metrics ──────────────────────────────",
            f"  Total Trades        : {analysis.total_trades}",
            f"  Win Rate            : {analysis.win_rate:.1%}   "
            f"{'✅' if analysis.meets_win_rate else '❌'} (target >55%)",
            f"  Profit Factor       : {analysis.profit_factor:.2f}  "
            f"{'✅' if analysis.meets_profit_factor else '❌'} (target >1.5)",
            f"  Sharpe Ratio        : {analysis.sharpe_ratio:.2f}  "
            f"{'✅' if analysis.meets_sharpe else '❌'} (target >1.0)",
            f"  Max Drawdown        : {analysis.max_drawdown_pct:.1f}%  "
            f"{'✅' if analysis.meets_drawdown else '❌'} (target <20%)",
            f"  Recovery Factor     : {analysis.recovery_factor:.2f}  "
            f"{'✅' if analysis.meets_recovery else '❌'} (target >2.0)",
            "",
            "── Return Metrics ───────────────────────────────────",
            f"  Total Return        : {analysis.total_return_pct:.2f}%",
            f"  Annual Return       : {analysis.annual_return_pct:.2f}%",
            f"  Sortino Ratio       : {analysis.sortino_ratio:.2f}",
            "",
            "── Trade Statistics ─────────────────────────────────",
            f"  Avg Win             : {analysis.avg_win_pct:.2f}%",
            f"  Avg Loss            : {analysis.avg_loss_pct:.2f}%",
            f"  Largest Win         : {analysis.largest_win_pct:.2f}%",
            f"  Largest Loss        : {analysis.largest_loss_pct:.2f}%",
            f"  Max Consec. Wins    : {analysis.consecutive_wins}",
            f"  Max Consec. Losses  : {analysis.consecutive_losses}",
            "",
        ]

        if walk_forward:
            lines += [
                "── Walk-Forward Validation ──────────────────────────",
                f"  Windows Tested      : {walk_forward.n_windows}",
                f"  Avg Win Rate        : {walk_forward.avg_win_rate:.1%}",
                f"  Avg Profit Factor   : {walk_forward.avg_profit_factor:.2f}",
                f"  Avg Sharpe Ratio    : {walk_forward.avg_sharpe_ratio:.2f}",
                f"  Avg Max Drawdown    : {walk_forward.avg_max_drawdown:.1f}%",
                f"  Strategy Robust     : {'✅ YES' if walk_forward.is_robust else '❌ NO'}",
                "",
            ]

        overall = "✅ PASS — Strategy meets all success criteria" if analysis.overall_pass \
            else "❌ FAIL — Strategy does not meet all success criteria"
        lines += [
            "── Overall Validation ───────────────────────────────",
            f"  {overall}",
        ]
        if analysis.issues:
            lines.append("  Issues:")
            for issue in analysis.issues:
                lines.append(f"    • {issue}")
        lines.append(_DIVIDER)

        return "\n".join(lines)

    def generate_summary_dict(self, analysis: BacktestAnalysis) -> dict:
        """Return the analysis as a plain dict (for JSON API responses)."""
        return asdict(analysis)
