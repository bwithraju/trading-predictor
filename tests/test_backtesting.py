"""Tests for backtesting engine and analyzer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine, BacktestResult, TradeLog
from src.backtesting.backtest_analyzer import BacktestAnalyzer
from src.backtesting.walk_forward import WalkForwardAnalyzer
from src.backtesting.report_generator import ReportGenerator


class TestBacktestEngine:
    def test_run_returns_result(self, long_ohlcv_df):
        engine = BacktestEngine(lookback=50)
        result = engine.run(long_ohlcv_df, "TEST", risk_percent=2.0, account_size=10_000)
        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert result.total_trades >= 0
        assert 0.0 <= result.win_rate <= 1.0
        assert result.profit_factor >= 0.0
        assert result.max_drawdown_pct >= 0.0

    def test_empty_df_returns_zero_trades(self):
        engine = BacktestEngine()
        result = engine.run(pd.DataFrame(), "EMPTY")
        assert result.total_trades == 0

    def test_small_df(self, short_ohlcv_df):
        engine = BacktestEngine(lookback=50)
        result = engine.run(short_ohlcv_df, "SMALL")
        assert result.total_trades == 0

    def test_compute_metrics_no_trades(self):
        result = BacktestEngine._compute_metrics("X", [], [10_000.0], 10_000.0)
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0


class TestBacktestAnalyzer:
    def _make_result(self, wins: int, losses: int) -> BacktestResult:
        trades = []
        for i in range(wins):
            trades.append(TradeLog(
                entry_index=i, entry_price=100.0, direction="LONG",
                stop_loss=98.0, take_profit=104.0, exit_price=104.0,
                exit_index=i + 1, outcome="WIN", pnl_pct=4.0,
            ))
        for i in range(losses):
            idx = wins + i
            trades.append(TradeLog(
                entry_index=idx, entry_price=100.0, direction="LONG",
                stop_loss=98.0, take_profit=104.0, exit_price=98.0,
                exit_index=idx + 1, outcome="LOSS", pnl_pct=-2.0,
            ))

        equity = [10_000.0]
        for t in trades:
            equity.append(equity[-1] * (1 + t.pnl_pct / 100))

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        gross_profit = wins * 4.0
        gross_loss = losses * 2.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_ret = (equity[-1] - equity[0]) / equity[0] * 100
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / peak * 100

        return BacktestResult(
            symbol="TEST", total_trades=wins + losses,
            win_trades=wins, loss_trades=losses,
            win_rate=round(win_rate, 4),
            total_return_pct=round(total_ret, 2),
            profit_factor=round(pf, 4),
            max_drawdown_pct=round(dd.max(), 2),
            sharpe_ratio=1.2,
            trades=trades,
        )

    def test_analyze_passing_strategy(self):
        result = self._make_result(wins=70, losses=30)
        analyzer = BacktestAnalyzer()
        analysis = analyzer.analyze(result)
        assert analysis.total_trades == 100
        assert analysis.win_rate == pytest.approx(0.7, abs=0.01)

    def test_analyze_failing_strategy(self):
        result = self._make_result(wins=30, losses=70)
        analyzer = BacktestAnalyzer()
        analysis = analyzer.analyze(result)
        assert not analysis.meets_win_rate
        assert not analysis.overall_pass
        assert len(analysis.issues) > 0

    def test_consecutive_streak(self):
        win_log = TradeLog(1, 100.0, "LONG", 98.0, 104.0, 104.0, 2, "WIN", 4.0)
        loss_log = TradeLog(3, 100.0, "LONG", 98.0, 104.0, 98.0, 4, "LOSS", -2.0)
        trades = [win_log, win_log, win_log, loss_log, win_log]
        assert BacktestAnalyzer._max_consecutive(trades, "WIN") == 3
        assert BacktestAnalyzer._max_consecutive(trades, "LOSS") == 1


class TestWalkForwardAnalyzer:
    def test_insufficient_data(self, short_ohlcv_df):
        wf = WalkForwardAnalyzer(n_windows=4)
        result = wf.run(short_ohlcv_df, "TEST")
        assert result.n_windows == 0

    def test_runs_on_larger_data(self, long_ohlcv_df):
        wf = WalkForwardAnalyzer(n_windows=2)
        result = wf.run(long_ohlcv_df, "TEST")
        assert result.n_windows >= 0


class TestReportGenerator:
    def test_generate_report(self):
        from src.backtesting.backtest_analyzer import BacktestAnalysis
        analysis = BacktestAnalysis(
            symbol="AAPL", start_date="2023-01-01", end_date="2026-04-05",
            total_trades=100, win_rate=0.60, profit_factor=1.8,
            max_drawdown_pct=12.0, sharpe_ratio=1.2, sortino_ratio=1.5,
            total_return_pct=35.0, annual_return_pct=11.0,
            avg_win_pct=3.0, avg_loss_pct=-2.0,
            largest_win_pct=8.0, largest_loss_pct=-5.0,
            consecutive_wins=7, consecutive_losses=3,
            recovery_factor=2.9, meets_win_rate=True,
            meets_profit_factor=True, meets_sharpe=True,
            meets_drawdown=True, meets_recovery=True, overall_pass=True,
        )
        reporter = ReportGenerator()
        report = reporter.generate_backtest_report(analysis)
        assert "AAPL" in report
        assert "PASS" in report

    def test_generate_summary_dict(self):
        from src.backtesting.backtest_analyzer import BacktestAnalysis
        analysis = BacktestAnalysis(
            symbol="MSFT", start_date="2023-01-01", end_date="2026-04-05",
            total_trades=50, win_rate=0.5, profit_factor=1.2,
            max_drawdown_pct=25.0, sharpe_ratio=0.8, sortino_ratio=1.0,
            total_return_pct=10.0, annual_return_pct=3.3,
            avg_win_pct=2.0, avg_loss_pct=-2.5,
            largest_win_pct=5.0, largest_loss_pct=-6.0,
            consecutive_wins=3, consecutive_losses=5,
            recovery_factor=0.4,
        )
        reporter = ReportGenerator()
        d = reporter.generate_summary_dict(analysis)
        assert isinstance(d, dict)
        assert d["symbol"] == "MSFT"
