"""Tests for paper trading modules."""
from __future__ import annotations

import pytest

from src.paper_trading.paper_account import PaperAccount, PaperPosition, ClosedTrade
from src.paper_trading.paper_engine import PaperEngine
from src.paper_trading.paper_tracker import PaperTracker


class TestPaperAccount:
    def test_initial_state(self):
        acct = PaperAccount(initial_capital=10_000.0, max_open_positions=5)
        assert acct.cash == 10_000.0
        assert acct.equity == 10_000.0
        assert len(acct.positions) == 0

    def test_open_position(self):
        acct = PaperAccount(initial_capital=10_000.0)
        success = acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        assert success
        assert "AAPL" in acct.positions
        assert acct.cash < 10_000.0

    def test_open_position_insufficient_cash(self):
        acct = PaperAccount(initial_capital=100.0)
        success = acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        assert not success
        assert "AAPL" not in acct.positions

    def test_open_position_max_limit(self):
        acct = PaperAccount(initial_capital=100_000.0, max_open_positions=2)
        acct.open_position("AAPL", "LONG", 1.0, 150.0, 145.0, 160.0)
        acct.open_position("MSFT", "LONG", 1.0, 200.0, 195.0, 210.0)
        success = acct.open_position("TSLA", "LONG", 1.0, 300.0, 290.0, 320.0)
        assert not success

    def test_close_position_win(self):
        acct = PaperAccount(initial_capital=10_000.0, commission_rate=0.0)
        acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        trade = acct.close_position("AAPL", 160.0, "TP_HIT")
        assert trade is not None
        assert trade.outcome == "WIN"
        assert trade.pnl > 0

    def test_close_position_loss(self):
        acct = PaperAccount(initial_capital=10_000.0, commission_rate=0.0)
        acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        trade = acct.close_position("AAPL", 145.0, "SL_HIT")
        assert trade is not None
        assert trade.outcome == "LOSS"
        assert trade.pnl < 0

    def test_close_nonexistent_position(self):
        acct = PaperAccount()
        trade = acct.close_position("FAKE", 100.0)
        assert trade is None

    def test_update_prices(self):
        acct = PaperAccount(initial_capital=10_000.0, commission_rate=0.0)
        acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        acct.update_prices({"AAPL": 155.0})
        assert acct.positions["AAPL"].current_price == 155.0
        assert acct.equity > 10_000.0 - 1_500.0  # rough check

    def test_unrealized_pnl_short(self):
        acct = PaperAccount(initial_capital=100_000.0, commission_rate=0.0)
        acct.open_position("TSLA", "SHORT", 10.0, 300.0, 310.0, 280.0)
        acct.update_prices({"TSLA": 290.0})
        pos = acct.positions["TSLA"]
        assert pos.unrealized_pnl > 0  # price fell, SHORT is profitable

    def test_reset(self):
        acct = PaperAccount(initial_capital=10_000.0)
        acct.open_position("AAPL", "LONG", 1.0, 150.0, 145.0, 160.0)
        acct.reset()
        assert acct.cash == 10_000.0
        assert len(acct.positions) == 0


class TestPaperEngine:
    def test_start_stop(self):
        engine = PaperEngine()
        assert not engine.is_running
        engine.start()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_execute_signal_not_running(self):
        engine = PaperEngine()
        success = engine.execute_signal("AAPL", "LONG", 150.0, 145.0, 160.0, 10.0)
        assert not success

    def test_execute_signal_running(self):
        engine = PaperEngine(account=PaperAccount(initial_capital=100_000.0, commission_rate=0.0))
        engine.start()
        success = engine.execute_signal("AAPL", "LONG", 150.0, 145.0, 160.0, 10.0)
        assert success

    def test_process_bar_sl_hit(self):
        engine = PaperEngine(account=PaperAccount(initial_capital=100_000.0, commission_rate=0.0))
        engine.start()
        engine.execute_signal("AAPL", "LONG", 150.0, 145.0, 160.0, 10.0)
        result = engine.process_bar("AAPL", high=149.0, low=144.0, close=144.5)
        assert result == "SL_HIT"
        assert "AAPL" not in engine.account.positions

    def test_process_bar_tp_hit(self):
        engine = PaperEngine(account=PaperAccount(initial_capital=100_000.0, commission_rate=0.0))
        engine.start()
        engine.execute_signal("AAPL", "LONG", 150.0, 145.0, 160.0, 10.0)
        result = engine.process_bar("AAPL", high=161.0, low=155.0, close=160.5)
        assert result == "TP_HIT"

    def test_process_bar_no_exit(self):
        engine = PaperEngine(account=PaperAccount(initial_capital=100_000.0, commission_rate=0.0))
        engine.start()
        engine.execute_signal("AAPL", "LONG", 150.0, 145.0, 160.0, 10.0)
        result = engine.process_bar("AAPL", high=155.0, low=149.0, close=153.0)
        assert result is None


class TestPaperTracker:
    def _setup_with_trades(self) -> tuple[PaperAccount, PaperTracker]:
        acct = PaperAccount(initial_capital=10_000.0, commission_rate=0.0)
        tracker = PaperTracker(acct)
        # Open and close a few trades
        acct.open_position("AAPL", "LONG", 10.0, 150.0, 145.0, 160.0)
        acct.close_position("AAPL", 160.0, "TP_HIT")  # WIN
        acct.open_position("MSFT", "LONG", 5.0, 200.0, 195.0, 210.0)
        acct.close_position("MSFT", 195.0, "SL_HIT")  # LOSS
        return acct, tracker

    def test_compute_performance(self):
        _, tracker = self._setup_with_trades()
        perf = tracker.compute_performance()
        assert perf.total_trades == 2
        assert perf.win_trades == 1
        assert perf.loss_trades == 1
        assert perf.win_rate == 0.5

    def test_get_trade_journal(self):
        _, tracker = self._setup_with_trades()
        journal = tracker.get_trade_journal()
        assert len(journal) == 2
        symbols = {t["symbol"] for t in journal}
        assert "AAPL" in symbols
        assert "MSFT" in symbols
