"""FastAPI leverage prediction endpoints.

Routes:
    POST /leverage/recommend
    GET  /leverage/status
    POST /leverage/adjust
    GET  /leverage/analysis/{symbol}
    POST /leverage/backtest
    GET  /leverage/insights

Mount this router in your main FastAPI app::

    from src.api.leverage_endpoints import router as leverage_router
    app.include_router(leverage_router)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field, field_validator

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

from src.leverage.calculator import LeverageCalculator
from src.leverage.model import LeverageModel
from src.leverage.safety_checker import SafetyChecker
from src.leverage.tiers import TIER_REGISTRY, VolatilityTier, classify_volatility_tier

# ---------------------------------------------------------------------------
# Shared model instance (singleton per process)
# ---------------------------------------------------------------------------

_leverage_model: Optional[LeverageModel] = None


def get_leverage_model() -> LeverageModel:
    global _leverage_model
    if _leverage_model is None:
        _leverage_model = LeverageModel()
        _leverage_model.load_model()  # no-op if model files absent
    return _leverage_model


if _FASTAPI_AVAILABLE:

    router = APIRouter(prefix="/leverage", tags=["Leverage"])

    # -----------------------------------------------------------------------
    # Pydantic schemas
    # -----------------------------------------------------------------------

    class RecommendRequest(BaseModel):
        symbol: str = Field(..., json_schema_extra={"example": "BTC/USDT"})
        timeframe: str = Field("1h", json_schema_extra={"example": "1h"})
        account_size: float = Field(..., gt=0, json_schema_extra={"example": 1000.0})
        close_prices: List[float] = Field(..., min_length=30, description="Close prices oldest-first")
        high_prices: List[float] = Field(..., min_length=30, description="High prices oldest-first")
        low_prices: List[float] = Field(..., min_length=30, description="Low prices oldest-first")
        risk_percent: float = Field(2.0, gt=0, le=100)
        stop_loss_percent: float = Field(5.0, gt=0, le=100)

        @field_validator("high_prices", "low_prices")
        @classmethod
        def same_length_as_close(cls, v: list, info) -> list:
            close = info.data.get("close_prices")
            if close is not None and len(v) != len(close):
                raise ValueError("high_prices and low_prices must have the same length as close_prices")
            return v

    class RecommendResponse(BaseModel):
        symbol: str
        timeframe: str
        recommended_leverage: int
        max_safe_leverage: int
        aggressive_leverage: int
        confidence_score: float
        safety_score: float
        liquidation_risk_pct: float
        volatility_tier: str
        volatility_ratio: float
        trend_direction: int
        trend_strength: float
        account_size: float
        buying_power: float
        max_position_size: float
        daily_stop_loss: float
        risk_level: str
        adjustment_needed: bool
        adjustment_reason: str
        safety_report: dict

    class StatusResponse(BaseModel):
        model_loaded: bool
        available_tiers: List[str]
        supported_leverage_range: Dict[str, int]

    class AdjustRequest(BaseModel):
        current_leverage: int = Field(..., ge=1, le=20)
        new_leverage_request: int = Field(..., ge=1, le=20)
        account_size: float = Field(..., gt=0)
        current_drawdown_pct: float = Field(0.0, ge=0, le=100)
        volatility_ratio: float = Field(1.0, gt=0)

    class AdjustResponse(BaseModel):
        approved: bool
        reason: str
        safer_alternative: int
        liquidation_risk_pct: float

    class AnalysisResponse(BaseModel):
        symbol: str
        volatility_tier: str
        volatility_ratio: float
        max_leverage_for_tier: int
        tier_description: str
        liquidation_risk_range: List[float]
        features: Dict

    class BacktestRequest(BaseModel):
        close_prices: List[float] = Field(..., min_length=100)
        high_prices: List[float] = Field(..., min_length=100)
        low_prices: List[float] = Field(..., min_length=100)
        strategy: str = Field("dynamic", pattern="^(dynamic|fixed)$")
        fixed_leverage: Optional[int] = Field(None, ge=1, le=20)
        account_size: float = Field(1000.0, gt=0)

    class BacktestResponse(BaseModel):
        strategy: str
        total_bars: int
        avg_recommended_leverage: float
        leverage_distribution: Dict[str, int]
        avg_liquidation_risk_pct: float
        volatility_tier_distribution: Dict[str, int]
        estimated_safety_score: float

    class InsightsResponse(BaseModel):
        market_overview: str
        tier_thresholds: List[Dict]
        recommendations_summary: str
        risk_summary: str

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @router.post("/recommend", response_model=RecommendResponse)
    def recommend_leverage(request: RecommendRequest) -> RecommendResponse:
        """Generate a leverage recommendation for the given market conditions.

        - **symbol**: Trading pair, e.g. ``BTC/USDT``
        - **account_size**: Equity in USD
        - **close_prices / high_prices / low_prices**: OHLCV arrays (oldest first, ≥30 bars)
        """
        model = get_leverage_model()
        model.risk_percent = request.risk_percent
        model.stop_loss_percent = request.stop_loss_percent

        try:
            rec = model.recommend(
                close=np.array(request.close_prices),
                high=np.array(request.high_prices),
                low=np.array(request.low_prices),
                symbol=request.symbol,
                timeframe=request.timeframe,
                account_size=request.account_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return RecommendResponse(**rec.to_dict())

    @router.get("/status", response_model=StatusResponse)
    def get_leverage_status() -> StatusResponse:
        """Return current model status and supported leverage range."""
        model = get_leverage_model()
        return StatusResponse(
            model_loaded=model._model_loaded,
            available_tiers=[t.tier.value for t in TIER_REGISTRY],
            supported_leverage_range={"min": 1, "max": 20},
        )

    @router.post("/adjust", response_model=AdjustResponse)
    def adjust_leverage(request: AdjustRequest) -> AdjustResponse:
        """Approve or reject a requested leverage change.

        Returns a safer alternative when the request is rejected.
        """
        checker = SafetyChecker()
        calc = LeverageCalculator()

        tier_info = classify_volatility_tier(max(request.volatility_ratio, 1e-9))
        max_for_tier = tier_info.max_leverage

        # Drawdown factor: high drawdown → reduce cap
        drawdown_fraction = request.current_drawdown_pct / 100.0
        if drawdown_fraction > 0.20:
            max_for_tier = max(1, max_for_tier - 2)
        elif drawdown_fraction > 0.10:
            max_for_tier = max(1, max_for_tier - 1)

        safer_alternative = min(request.new_leverage_request, max_for_tier)

        liq_risk = calc.liquidation_risk(
            volatility=request.volatility_ratio * 0.3,  # rough mapping
            trend_strength=0.5,
            leverage=request.new_leverage_request,
        )

        approved = request.new_leverage_request <= max_for_tier

        if approved:
            reason = (
                f"Leverage {request.new_leverage_request}× approved for "
                f"{tier_info.tier.value} volatility conditions"
            )
        else:
            reason = (
                f"Leverage {request.new_leverage_request}× exceeds safe limit "
                f"{max_for_tier}× for {tier_info.tier.value} conditions. "
                f"Suggested: {safer_alternative}×"
            )

        return AdjustResponse(
            approved=approved,
            reason=reason,
            safer_alternative=safer_alternative,
            liquidation_risk_pct=round(liq_risk, 2),
        )

    @router.get("/analysis/{symbol}", response_model=AnalysisResponse)
    def get_leverage_analysis(
        symbol: str,
        close_prices: str = "50000,51000,50500,52000,51500",
    ) -> AnalysisResponse:
        """Return volatility tier and market analysis for a symbol.

        Pass ``close_prices`` as a comma-separated string of recent prices
        (minimum 30 values; up to 200 for best accuracy).  High/Low are
        approximated as ±0.5 % of close when not provided.
        """
        try:
            prices = [float(p) for p in close_prices.split(",") if p.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid close_prices: {exc}") from exc

        if len(prices) < 30:
            raise HTTPException(
                status_code=422,
                detail=f"Need at least 30 close prices, got {len(prices)}",
            )

        close = np.array(prices)
        high = close * 1.005
        low = close * 0.995

        model = get_leverage_model()
        result = model.analyze(close=close, high=high, low=low, symbol=symbol)
        result["liquidation_risk_range"] = list(result["liquidation_risk_range"])
        return AnalysisResponse(**result)

    @router.post("/backtest", response_model=BacktestResponse)
    def backtest_leverage(request: BacktestRequest) -> BacktestResponse:
        """Run a simplified leverage backtest over historical price data.

        Slides a window across the price series, computes a recommendation
        at each step, and aggregates the results.
        """
        close = np.array(request.close_prices)
        high = np.array(request.high_prices)
        low = np.array(request.low_prices)
        model = get_leverage_model()

        window = 50
        step = 5
        leverages: list[int] = []
        liq_risks: list[float] = []
        tiers: list[str] = []

        for start in range(0, len(close) - window, step):
            seg_c = close[start: start + window]
            seg_h = high[start: start + window]
            seg_l = low[start: start + window]

            try:
                rec = model.recommend(
                    close=seg_c,
                    high=seg_h,
                    low=seg_l,
                    symbol="BACKTEST",
                    timeframe="1h",
                    account_size=request.account_size,
                )
                lev = rec.recommended_leverage if request.strategy == "dynamic" else (request.fixed_leverage or 1)
                leverages.append(lev)
                liq_risks.append(rec.liquidation_risk_pct)
                tiers.append(rec.volatility_tier.value)
            except Exception:
                continue

        if not leverages:
            raise HTTPException(status_code=422, detail="No valid backtest windows computed")

        from collections import Counter

        lev_dist = dict(Counter(str(l) for l in leverages))
        tier_dist = dict(Counter(tiers))

        return BacktestResponse(
            strategy=request.strategy,
            total_bars=len(leverages),
            avg_recommended_leverage=round(float(np.mean(leverages)), 2),
            leverage_distribution=lev_dist,
            avg_liquidation_risk_pct=round(float(np.mean(liq_risks)), 2),
            volatility_tier_distribution=tier_dist,
            estimated_safety_score=round(max(0.0, 100.0 - float(np.mean(liq_risks))), 2),
        )

    @router.get("/insights", response_model=InsightsResponse)
    def get_leverage_insights() -> InsightsResponse:
        """Return market overview and leverage recommendations summary."""
        tier_thresholds = [
            {
                "tier": t.tier.value,
                "description": t.description,
                "volatility_ratio_min": t.volatility_ratio_min,
                "volatility_ratio_max": (
                    t.volatility_ratio_max if t.volatility_ratio_max != float("inf") else 999
                ),
                "max_leverage": t.max_leverage,
                "risk_label": t.risk_label,
                "liquidation_risk_range": list(t.liquidation_risk_range),
            }
            for t in TIER_REGISTRY
        ]

        return InsightsResponse(
            market_overview=(
                "Leverage recommendations are computed in real-time based on "
                "volatility tier, trend strength, and account-specific risk parameters."
            ),
            tier_thresholds=tier_thresholds,
            recommendations_summary=(
                "Use /leverage/recommend with live OHLCV data for personalised leverage advice."
            ),
            risk_summary=(
                "Always use stop-losses. Leverage amplifies both gains and losses. "
                "Start with lower leverage and increase only in low-volatility conditions."
            ),
        )
