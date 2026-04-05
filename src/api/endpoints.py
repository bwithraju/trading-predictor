"""FastAPI route definitions."""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import config
from src.analysis.technical import TechnicalAnalysis
from src.backtesting.engine import BacktestEngine
from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.models.predictor import TradingPredictor
from src.models.trainer import ModelTrainer
from src.risk.calculator import RiskCalculator
from src.utils.logger import get_logger
from src.utils.validators import validate_price, validate_risk_percent, validate_symbol

logger = get_logger(__name__)
router = APIRouter()

_fetcher = DataFetcher()
_cache = DataCache()
_ta = TechnicalAnalysis()
_predictor = TradingPredictor()


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
    asset_type: str = Field(default="stock")  # 'stock' or 'crypto'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = Field(default="2y")
    risk_percent: float = Field(default=2.0, gt=0, le=100)
    account_size: float = Field(default=10_000.0, gt=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(symbol: str, asset_type: str, period: str = None, interval: str = None):
    """Load data from cache or fetch fresh data."""
    if _cache.is_valid(symbol, asset_type):
        df = _cache.load(symbol, asset_type)
        if not df.empty:
            return df

    if asset_type == "crypto":
        df = _fetcher.fetch_crypto_data(symbol)
    else:
        df = _fetcher.fetch_stock_data(symbol, period=period, interval=interval)

    if not df.empty:
        _cache.save(df, symbol, asset_type)
    return df


# ---------------------------------------------------------------------------
# Endpoints
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
        df=df,
        symbol=symbol,
        entry_price=req.entry_price,
        risk_percent=req.risk_percent,
        account_size=req.account_size,
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
        df=df,
        symbol=symbol,
        entry_price=req.entry_price,
        risk_percent=req.risk_percent,
        account_size=req.account_size,
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
            entry_price=req.entry_price,
            exit_price=req.exit_price,
            direction=req.direction,
            account_size=req.account_size,
            risk_percent=req.risk_percent,
        )
    else:
        risk = calc.calculate(
            entry_price=req.entry_price,
            direction=req.direction,
            risk_percent=req.risk_percent,
            account_size=req.account_size,
        )
    return asdict(risk)


@router.post("/backtest", tags=["Backtesting"])
def backtest(req: BacktestRequest):
    """Run a backtest on historical data and return performance metrics."""
    try:
        symbol = validate_symbol(req.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if req.asset_type == "crypto":
        df = _fetcher.fetch_crypto_data(symbol)
    else:
        df = _fetcher.fetch_stock_data(
            symbol,
            period=req.period,
            start=req.start_date,
            end=req.end_date,
        )

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    engine = BacktestEngine()
    result = engine.run(
        df=df,
        symbol=symbol,
        risk_percent=req.risk_percent,
        account_size=req.account_size,
    )
    # Omit per-trade detail for a clean JSON response
    out = asdict(result)
    out.pop("trades", None)
    return out


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
