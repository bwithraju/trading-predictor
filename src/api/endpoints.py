"""FastAPI route definitions."""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import config
from src.analysis.technical import TechnicalAnalysis
from src.backtesting.backtest_analyzer import BacktestAnalyzer
from src.backtesting.engine import BacktestEngine
from src.backtesting.report_generator import ReportGenerator
from src.backtesting.walk_forward import WalkForwardAnalyzer
from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.models.predictor import TradingPredictor
from src.models.trainer import ModelTrainer
from src.paper_trading.paper_account import PaperAccount
from src.paper_trading.paper_engine import PaperEngine
from src.paper_trading.paper_tracker import PaperTracker
from src.risk.calculator import RiskCalculator
from src.trade_management.risk_monitor import RiskMonitor
from src.trade_management.trade_analyzer import TradeAnalyzer
from src.trade_management.trade_logger import TradeEntry, TradeLogger
from src.utils.logger import get_logger
from src.utils.validators import validate_price, validate_risk_percent, validate_symbol

logger = get_logger(__name__)
router = APIRouter()

_fetcher = DataFetcher()
_cache = DataCache()
_ta = TechnicalAnalysis()
_predictor = TradingPredictor()

# Paper trading state (in-process singleton)
_paper_account = PaperAccount()
_paper_engine = PaperEngine(account=_paper_account)
_paper_tracker = PaperTracker(account=_paper_account)
_trade_logger = TradeLogger()
_risk_monitor = RiskMonitor()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    symbol: str
    entry_price: float = Field(..., gt=0)
    risk_percent: float = Field(default=2.0, gt=0, le=100)
    account_size: float = Field(default=10_000.0, gt=0)
    period: Optional[str] = None
    interval: Optional[str] = None


class RiskRequest(BaseModel):
    entry_price: float = Field(..., gt=0)
    exit_price: Optional[float] = Field(default=None, gt=0)
    direction: str = Field(default="LONG")
    risk_percent: float = Field(default=2.0, gt=0, le=100)
    account_size: float = Field(default=10_000.0, gt=0)
    reward_ratio: float = Field(default=2.0, gt=0)


class BacktestRequest(BaseModel):
    symbol: str
    asset_type: str = Field(default="stock")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = Field(default="2y")
    risk_percent: float = Field(default=2.0, gt=0, le=100)
    account_size: float = Field(default=10_000.0, gt=0)
    run_walk_forward: bool = False


class PaperTradeRequest(BaseModel):
    symbol: str
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    qty: float = Field(..., gt=0)
    asset_type: str = Field(default="stock")
    entry_reason: str = Field(default="signal")


class LiveTradeRequest(BaseModel):
    symbol: str
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    qty: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(symbol: str, asset_type: str, period: str = None, interval: str = None,
               start: str = None, end: str = None):
    if not start and _cache.is_valid(symbol, asset_type):
        df = _cache.load(symbol, asset_type)
        if not df.empty:
            return df

    if asset_type == "crypto":
        df = _fetcher.fetch_crypto_data(symbol)
    else:
        df = _fetcher.fetch_stock_data(symbol, period=period, interval=interval,
                                       start=start, end=end)

    if not df.empty:
        _cache.save(df, symbol, asset_type)
    return df


# ---------------------------------------------------------------------------
# Original prediction / analysis endpoints
# ---------------------------------------------------------------------------


@router.post("/predict/stock", tags=["Predictions"])
def predict_stock(req: PredictRequest):
    """Predict stock direction with SL / TP / ML risk metrics."""
    try:
        symbol = validate_symbol(req.symbol)
        validate_price(req.entry_price)
        validate_risk_percent(req.risk_percent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    df = _load_data(symbol, "stock", period=req.period, interval=req.interval)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    result = _predictor.predict(
        df=df, symbol=symbol, entry_price=req.entry_price,
        risk_percent=req.risk_percent, account_size=req.account_size,
    )
    return asdict(result)


@router.post("/predict/crypto", tags=["Predictions"])
def predict_crypto(req: PredictRequest):
    """Predict crypto direction with SL / TP / ML risk metrics."""
    try:
        symbol = validate_symbol(req.symbol)
        validate_price(req.entry_price)
        validate_risk_percent(req.risk_percent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    df = _load_data(symbol, "crypto")
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    result = _predictor.predict(
        df=df, symbol=symbol, entry_price=req.entry_price,
        risk_percent=req.risk_percent, account_size=req.account_size,
    )
    return asdict(result)


@router.get("/analyze/{symbol}", tags=["Analysis"])
def analyze_symbol(symbol: str, asset_type: str = "stock"):
    """Return full technical analysis for a symbol."""
    try:
        symbol = validate_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    df = _load_data(symbol, asset_type)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    indicators = _ta.get_current_indicators(df)
    return {"symbol": symbol, "asset_type": asset_type, "indicators": indicators}


@router.get("/indicators/{symbol}", tags=["Analysis"])
def get_indicators(symbol: str, asset_type: str = "stock"):
    """Return the current indicator values for a symbol."""
    return analyze_symbol(symbol, asset_type)


@router.post("/calculate-risk", tags=["Risk"])
def calculate_risk(req: RiskRequest):
    """Calculate SL, TP, max-loss and position size for given parameters."""
    try:
        validate_price(req.entry_price)
        if req.exit_price:
            validate_price(req.exit_price, "exit_price")
        validate_risk_percent(req.risk_percent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    calc = RiskCalculator(reward_ratio=req.reward_ratio)
    if req.exit_price:
        risk = calc.calculate_from_exit(
            entry_price=req.entry_price, exit_price=req.exit_price,
            direction=req.direction, account_size=req.account_size,
            risk_percent=req.risk_percent,
        )
    else:
        risk = calc.calculate(
            entry_price=req.entry_price, direction=req.direction,
            risk_percent=req.risk_percent, account_size=req.account_size,
        )
    return asdict(risk)


@router.post("/backtest", tags=["Backtesting"])
def backtest(req: BacktestRequest):
    """Run a backtest on historical data and return performance metrics."""
    try:
        symbol = validate_symbol(req.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    start = req.start_date or config.backtest.DEFAULT_START_DATE
    end = req.end_date or config.backtest.DEFAULT_END_DATE

    df = _load_data(symbol, req.asset_type, period=req.period, start=start, end=end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    engine = BacktestEngine()
    result = engine.run(df=df, symbol=symbol, risk_percent=req.risk_percent,
                        account_size=req.account_size)

    analyzer = BacktestAnalyzer()
    analysis = analyzer.analyze(result, start_date=start, end_date=end)

    wf_result = None
    if req.run_walk_forward:
        wf_analyzer = WalkForwardAnalyzer()
        wf_result = wf_analyzer.run(df=df, symbol=symbol, risk_percent=req.risk_percent,
                                    account_size=req.account_size)

    reporter = ReportGenerator()
    report_text = reporter.generate_backtest_report(analysis, wf_result)
    summary = reporter.generate_summary_dict(analysis)

    response = {
        "symbol": symbol,
        "start_date": start,
        "end_date": end,
        "summary": summary,
        "report": report_text,
    }
    if wf_result:
        response["walk_forward"] = {
            "n_windows": wf_result.n_windows,
            "avg_win_rate": wf_result.avg_win_rate,
            "avg_profit_factor": wf_result.avg_profit_factor,
            "avg_sharpe_ratio": wf_result.avg_sharpe_ratio,
            "is_robust": wf_result.is_robust,
        }
    return response


@router.get("/trading/results", tags=["Backtesting"])
def get_backtest_config():
    """Return default backtesting configuration and success criteria."""
    return {
        "default_start_date": config.backtest.DEFAULT_START_DATE,
        "default_end_date": config.backtest.DEFAULT_END_DATE,
        "stock_symbols": config.backtest.STOCK_SYMBOLS,
        "crypto_symbols": config.backtest.CRYPTO_SYMBOLS,
        "timeframes": config.backtest.TIMEFRAMES,
        "success_criteria": {
            "min_win_rate": config.metrics.MIN_WIN_RATE,
            "min_profit_factor": config.metrics.MIN_PROFIT_FACTOR,
            "min_sharpe_ratio": config.metrics.MIN_SHARPE_RATIO,
            "max_drawdown_pct": config.metrics.MAX_DRAWDOWN_PCT,
            "min_recovery_factor": config.metrics.MIN_RECOVERY_FACTOR,
        },
    }


@router.post("/trading/validate", tags=["Backtesting"])
def validate_strategy(req: BacktestRequest):
    """Run a full backtest + walk-forward and return a validation report."""
    req.run_walk_forward = True
    return backtest(req)


@router.post("/train/{symbol}", tags=["Models"])
def train_model(symbol: str, asset_type: str = "stock"):
    """Train the prediction model for *symbol* on its historical data."""
    try:
        symbol = validate_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    df = _load_data(symbol, asset_type)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    trainer = ModelTrainer()
    try:
        metrics = trainer.train(df, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"symbol": symbol, "metrics": metrics}


# ---------------------------------------------------------------------------
# Paper trading endpoints
# ---------------------------------------------------------------------------


@router.post("/trading/paper/start", tags=["Paper Trading"])
def start_paper_trading():
    """Start the paper trading engine."""
    _paper_engine.start()
    return {"status": "started", "account": _paper_account.get_summary()}


@router.post("/trading/paper/stop", tags=["Paper Trading"])
def stop_paper_trading():
    """Stop the paper trading engine."""
    _paper_engine.stop()
    perf = _paper_tracker.compute_performance()
    return {
        "status": "stopped",
        "account": _paper_account.get_summary(),
        "performance": asdict(perf),
    }


@router.get("/trading/paper/account", tags=["Paper Trading"])
def get_paper_account():
    """Return the current paper account status and performance."""
    perf = _paper_tracker.compute_performance()
    return {
        "account": _paper_account.get_summary(),
        "performance": asdict(perf),
        "engine_running": _paper_engine.is_running,
    }


@router.post("/trading/paper/trade", tags=["Paper Trading"])
def execute_paper_trade(req: PaperTradeRequest):
    """Execute a simulated paper trade."""
    if not _paper_engine.is_running:
        raise HTTPException(
            status_code=400, detail="Paper engine is not running. Start it first."
        )

    import uuid
    trade_id = str(uuid.uuid4())[:8]
    success = _paper_engine.execute_signal(
        symbol=req.symbol,
        direction=req.direction,
        entry_price=req.entry_price,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        position_size=req.qty,
    )

    if success:
        _trade_logger.log_entry(TradeEntry(
            trade_id=trade_id,
            symbol=req.symbol,
            direction=req.direction,
            asset_type=req.asset_type,
            entry_price=req.entry_price,
            qty=req.qty,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            entry_reason=req.entry_reason,
            mode="paper",
        ))
        return {
            "success": True,
            "trade_id": trade_id,
            "message": f"Paper trade opened for {req.symbol}",
            "account": _paper_account.get_summary(),
        }
    raise HTTPException(
        status_code=400,
        detail=f"Could not open paper position for {req.symbol}. "
               "Check position limits or available cash.",
    )


@router.post("/trading/paper/close/{symbol}", tags=["Paper Trading"])
def close_paper_position(symbol: str, exit_price: float):
    """Manually close a paper position."""
    trade = _paper_engine.close_position(symbol, exit_price)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"No open position found for {symbol}")
    return {
        "success": True,
        "symbol": symbol,
        "pnl": trade.pnl,
        "pnl_pct": trade.pnl_pct,
        "outcome": trade.outcome,
        "account": _paper_account.get_summary(),
    }


# ---------------------------------------------------------------------------
# Live trading endpoints (gated behind feature flag)
# ---------------------------------------------------------------------------


@router.post("/trading/live/execute", tags=["Live Trading"])
def execute_live_trade(req: LiveTradeRequest):
    """Execute a live trade via Alpaca (requires LIVE_TRADING_ENABLED=true)."""
    if not config.live.ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled. Set LIVE_TRADING_ENABLED=true to enable.",
        )

    from src.trading.alpaca_client import AlpacaClient
    from src.trading.order_manager import OrderManager

    client = AlpacaClient()
    if not client.connect():
        raise HTTPException(status_code=503, detail="Could not connect to Alpaca API")

    order_mgr = OrderManager(client)
    side = "buy" if req.direction == "LONG" else "sell"

    order = order_mgr.submit_bracket_order(
        symbol=req.symbol,
        qty=req.qty,
        side=side,
        limit_price=req.entry_price,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
    )

    if order is None:
        raise HTTPException(status_code=500, detail="Order submission failed")

    return {
        "success": True,
        "order_id": order.order_id,
        "symbol": req.symbol,
        "side": side,
        "qty": req.qty,
        "status": order.status,
    }


@router.get("/trading/positions", tags=["Live Trading"])
def get_positions():
    """Return current live positions (requires Alpaca credentials)."""
    if not config.live.ENABLED:
        # Return paper positions instead
        return {
            "mode": "paper",
            "positions": [
                {
                    "symbol": s,
                    "qty": p.qty,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                }
                for s, p in _paper_account.positions.items()
            ],
        }

    from src.trading.alpaca_client import AlpacaClient
    from src.trading.position_manager import PositionManager

    client = AlpacaClient()
    if not client.connect():
        raise HTTPException(status_code=503, detail="Could not connect to Alpaca API")

    pos_mgr = PositionManager(client)
    return {"mode": "live", **pos_mgr.get_position_summary()}


# ---------------------------------------------------------------------------
# Trade journal endpoints
# ---------------------------------------------------------------------------


@router.get("/trading/journal", tags=["Trade Journal"])
def get_trade_journal():
    """Return all trades from the trade journal."""
    return {
        "open_trades": _trade_logger.load_open_trades(),
        "closed_trades": _trade_logger.load_closed_trades(),
    }


@router.get("/trading/journal/performance", tags=["Trade Journal"])
def get_journal_performance():
    """Return performance analysis of all journaled trades."""
    analyzer = TradeAnalyzer(_trade_logger)
    return {
        "overall": analyzer.analyze_all(),
        "by_symbol": analyzer.analyze_by_symbol(),
        "by_mode": analyzer.analyze_by_mode(),
    }


@router.get("/trading/risk-status", tags=["Risk"])
def get_risk_status():
    """Return current risk monitoring status for paper trading session."""
    account_summary = _paper_account.get_summary()
    equity = account_summary["equity"]
    initial = account_summary["initial_capital"]
    open_pos = account_summary["open_positions"]

    risk_status = _risk_monitor.check_trade_allowed(
        current_equity=equity,
        start_of_day_equity=initial,
        open_position_count=open_pos,
        proposed_trade_size_pct=config.live.MAX_RISK_PER_TRADE_PCT,
    )
    return asdict(risk_status)
