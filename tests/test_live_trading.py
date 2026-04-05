"""Tests for the Alpaca live trading client (mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.trading.alpaca_client import AlpacaClient
from src.trading.order_manager import OrderManager, OrderRecord
from src.trading.position_manager import PositionManager, Position
from src.trading.account_manager import AccountManager


class TestAlpacaClient:
    def test_connect_no_credentials(self):
        client = AlpacaClient(api_key="", secret_key="", paper=True)
        result = client.connect()
        assert result is False
        assert not client.is_connected

    def test_connect_with_mock(self):
        mock_api = MagicMock()
        mock_api.get_account.return_value = MagicMock()

        with patch("alpaca_trade_api.REST", return_value=mock_api):
            client = AlpacaClient(api_key="test_key", secret_key="test_secret",
                                  base_url="https://paper-api.alpaca.markets", paper=True)
            result = client.connect()
            assert result is True
            assert client.is_connected

    def test_get_account_not_connected(self):
        client = AlpacaClient()
        result = client.get_account()
        assert result == {}

    def test_get_account_with_mock(self):
        mock_api = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "abc123"
        mock_account.status = "ACTIVE"
        mock_account.equity = "15000.00"
        mock_account.cash = "5000.00"
        mock_account.buying_power = "10000.00"
        mock_account.portfolio_value = "15000.00"
        mock_account.currency = "USD"
        mock_api.get_account.return_value = mock_account

        client = AlpacaClient(api_key="key", secret_key="secret", paper=True)
        client._api = mock_api

        result = client.get_account()
        assert result["equity"] == 15000.0
        assert result["status"] == "ACTIVE"
        assert result["paper"] is True


class TestOrderManager:
    def _make_client_with_api(self) -> tuple[AlpacaClient, MagicMock]:
        mock_api = MagicMock()
        client = AlpacaClient()
        client._api = mock_api
        return client, mock_api

    def test_submit_market_order(self):
        client, mock_api = self._make_client_with_api()
        mock_order = MagicMock()
        mock_order.id = "ord001"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.type = "market"
        mock_order.qty = "10"
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.status = "new"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_api.submit_order.return_value = mock_order

        mgr = OrderManager(client)
        record = mgr.submit_market_order("AAPL", 10, "buy")
        assert record is not None
        assert record.order_id == "ord001"
        assert record.symbol == "AAPL"

    def test_submit_order_not_connected(self):
        client = AlpacaClient()  # no api
        mgr = OrderManager(client)
        result = mgr.submit_market_order("AAPL", 10, "buy")
        assert result is None

    def test_get_order_history_empty(self):
        client = AlpacaClient()
        mgr = OrderManager(client)
        assert mgr.get_order_history() == []

    def test_submit_limit_order(self):
        client, mock_api = self._make_client_with_api()
        mock_order = MagicMock()
        mock_order.id = "ord002"
        mock_order.symbol = "MSFT"
        mock_order.side = "buy"
        mock_order.type = "limit"
        mock_order.qty = "5"
        mock_order.limit_price = "300.00"
        mock_order.stop_price = None
        mock_order.status = "new"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_api.submit_order.return_value = mock_order

        mgr = OrderManager(client)
        record = mgr.submit_limit_order("MSFT", 5, "buy", 300.0)
        assert record is not None
        assert record.limit_price == 300.0


class TestPositionManager:
    def test_get_all_positions_not_connected(self):
        client = AlpacaClient()
        mgr = PositionManager(client)
        assert mgr.get_all_positions() == []

    def test_get_all_positions_with_mock(self):
        mock_api = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.side = "long"
        mock_pos.avg_entry_price = "150.00"
        mock_pos.current_price = "155.00"
        mock_pos.market_value = "1550.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.unrealized_plpc = "0.0333"
        mock_api.list_positions.return_value = [mock_pos]

        client = AlpacaClient()
        client._api = mock_api
        mgr = PositionManager(client)
        positions = mgr.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    def test_calculate_position_size(self):
        client = AlpacaClient()
        mgr = PositionManager(client)
        size = mgr.calculate_position_size(
            account_size=10_000.0, entry_price=100.0, stop_loss=98.0,
            risk_percent=2.0, max_open_positions=5
        )
        # max risk = 200, sl_distance = 2.0 → raw = 100 shares
        # max allocation = 10000/5/100 = 20 shares
        assert size == pytest.approx(20.0, abs=0.1)


class TestAccountManager:
    def test_snapshot_not_connected(self):
        client = AlpacaClient()
        mgr = AccountManager(client)
        snap = mgr.get_snapshot()
        assert snap is None

    def test_daily_loss_not_breached(self):
        mock_api = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "x"
        mock_account.status = "ACTIVE"
        mock_account.equity = "10000.00"
        mock_account.cash = "5000.00"
        mock_account.buying_power = "10000.00"
        mock_account.portfolio_value = "10000.00"
        mock_account.currency = "USD"
        mock_api.get_account.return_value = mock_account

        client = AlpacaClient(paper=True)
        client._api = mock_api
        mgr = AccountManager(client, daily_loss_limit_pct=5.0)
        mgr._start_of_day_equity = 10_000.0

        assert not mgr.is_daily_loss_limit_breached()

    def test_portfolio_summary(self):
        client = AlpacaClient()
        mgr = AccountManager(client)
        summary = mgr.get_portfolio_summary()
        assert "error" in summary  # not connected
