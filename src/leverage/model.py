"""Leverage prediction model.

Provides :class:`LeverageModel` which combines:
  - Feature engineering (:class:`~src.leverage.features.LeverageFeatures`)
  - Volatility tier classification (:func:`~src.leverage.tiers.classify_volatility_tier`)
  - Scikit-learn ensemble (Random Forest + Gradient Boosting)
  - Safety checks (:class:`~src.leverage.safety_checker.SafetyChecker`)
  - Financial calculations (:class:`~src.leverage.calculator.LeverageCalculator`)

When no trained model is available the tier-based heuristic is used as a
fallback, which still provides valid recommendations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .calculator import LeverageCalculator
from .features import LeverageFeatures
from .safety_checker import SafetyChecker, SafetyReport
from .tiers import TierInfo, VolatilityTier, classify_volatility_tier

# Optional sklearn imports – gracefully absent in minimal environments
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# Optional joblib for model persistence
try:
    import joblib

    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JOBLIB_AVAILABLE = False

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class LeverageRecommendation:
    """Full leverage recommendation returned by :class:`LeverageModel`."""

    symbol: str
    timeframe: str

    # Core recommendation
    recommended_leverage: int
    max_safe_leverage: int
    aggressive_leverage: int

    # Scoring
    confidence_score: float  # 0–100
    safety_score: float  # 0–100
    liquidation_risk_pct: float

    # Market context
    volatility_tier: VolatilityTier
    volatility_ratio: float
    trend_direction: int  # +1 / 0 / -1
    trend_strength: float  # ADX normalised 0–1

    # Account info
    account_size: float
    buying_power: float
    max_position_size: float
    daily_stop_loss: float

    # Risk level label
    risk_level: str

    # Safety report
    safety_report: SafetyReport = field(repr=False)

    # Adjustment flags
    adjustment_needed: bool = False
    adjustment_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "recommended_leverage": self.recommended_leverage,
            "max_safe_leverage": self.max_safe_leverage,
            "aggressive_leverage": self.aggressive_leverage,
            "confidence_score": round(self.confidence_score, 2),
            "safety_score": round(self.safety_score, 2),
            "liquidation_risk_pct": round(self.liquidation_risk_pct, 2),
            "volatility_tier": self.volatility_tier.value,
            "volatility_ratio": round(self.volatility_ratio, 4),
            "trend_direction": self.trend_direction,
            "trend_strength": round(self.trend_strength, 4),
            "account_size": self.account_size,
            "buying_power": round(self.buying_power, 2),
            "max_position_size": round(self.max_position_size, 2),
            "daily_stop_loss": round(self.daily_stop_loss, 2),
            "risk_level": self.risk_level,
            "adjustment_needed": self.adjustment_needed,
            "adjustment_reason": self.adjustment_reason,
            "safety_report": self.safety_report.to_dict(),
        }


class LeverageModel:
    """Intelligent leverage prediction model.

    Usage::

        model = LeverageModel()
        # Optionally load a pre-trained sklearn model:
        # model.load_model()

        recommendation = model.recommend(
            close=close_prices,
            high=high_prices,
            low=low_prices,
            symbol="BTC/USDT",
            timeframe="1h",
            account_size=1000.0,
        )
        print(recommendation.recommended_leverage)
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        safety_checker: Optional[SafetyChecker] = None,
        calculator: Optional[LeverageCalculator] = None,
        features_engine: Optional[LeverageFeatures] = None,
        risk_percent: float = 2.0,
        stop_loss_percent: float = 5.0,
        daily_loss_limit_pct: float = 5.0,
    ) -> None:
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.safety_checker = safety_checker or SafetyChecker()
        self.calculator = calculator or LeverageCalculator()
        self.features_engine = features_engine or LeverageFeatures()
        self.risk_percent = risk_percent
        self.stop_loss_percent = stop_loss_percent
        self.daily_loss_limit_pct = daily_loss_limit_pct

        # Sklearn components (populated by load_model / train)
        self._clf: Optional[object] = None  # RandomForestClassifier (safe/risky)
        self._gbr: Optional[object] = None  # GradientBoostingRegressor (leverage magnitude)
        self._scaler: Optional[object] = None  # StandardScaler
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Attempt to load a pre-trained model from *model_dir*.

        Returns:
            True if loading succeeded, False otherwise.
        """
        if not _JOBLIB_AVAILABLE or not _SKLEARN_AVAILABLE:
            return False  # pragma: no cover

        clf_path = self.model_dir / "leverage_model.pkl"
        scaler_path = self.model_dir / "leverage_scaler.pkl"

        if not clf_path.exists() or not scaler_path.exists():
            return False

        try:
            self._clf = joblib.load(clf_path)
            self._scaler = joblib.load(scaler_path)
            self._model_loaded = True
            return True
        except Exception:
            self._model_loaded = False
            return False

    def save_model(self) -> bool:
        """Save trained model components to *model_dir*.

        Returns:
            True if saving succeeded.
        """
        if not _JOBLIB_AVAILABLE or not self._model_loaded:
            return False  # pragma: no cover

        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self._clf, self.model_dir / "leverage_model.pkl")
            joblib.dump(self._scaler, self.model_dir / "leverage_scaler.pkl")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Heuristic (tier-based) recommendation – always available
    # ------------------------------------------------------------------

    def _heuristic_recommendation(
        self,
        tier_info: TierInfo,
        adx: float,
        current_drawdown: float,
    ) -> tuple[int, float]:
        """Return (leverage, confidence) using tier + adjustment rules.

        Applies dynamic reduction rules:
          - High drawdown → reduce
          - Weak trend (low ADX) → reduce
        """
        leverage = tier_info.max_leverage
        confidence = 75.0  # base confidence for heuristic

        # Drawdown penalty
        if current_drawdown > 0.20:
            leverage = max(1, leverage - 2)
            confidence -= 15
        elif current_drawdown > 0.10:
            leverage = max(1, leverage - 1)
            confidence -= 7

        # Weak trend penalty
        adx_norm = min(adx / 100.0, 1.0) if not np.isnan(adx) else 0.3
        if adx_norm < 0.2:
            leverage = max(1, leverage - 1)
            confidence -= 5

        return leverage, max(10.0, min(confidence, 100.0))

    # ------------------------------------------------------------------
    # ML-based recommendation
    # ------------------------------------------------------------------

    def _ml_recommendation(self, feature_array: np.ndarray) -> tuple[int, float]:
        """Return (leverage, confidence) from the sklearn model."""
        if not self._model_loaded or self._clf is None:
            raise RuntimeError("Model not loaded")

        X = self._scaler.transform(feature_array.reshape(1, -1))

        # Safety classification probability
        safe_proba = float(self._clf.predict_proba(X)[0][1])  # P(safe)
        confidence = safe_proba * 100

        # Leverage magnitude regression
        if self._gbr is not None:
            raw_leverage = float(self._gbr.predict(X)[0])
        else:
            raw_leverage = safe_proba * 15 + 1  # fallback linear mapping

        leverage = int(round(max(1, min(raw_leverage, 20))))
        return leverage, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        account_size: float = 1000.0,
        volume: Optional[np.ndarray] = None,
    ) -> LeverageRecommendation:
        """Generate a full leverage recommendation.

        Args:
            close: Close prices (oldest first), at least 30 bars.
            high: High prices.
            low: Low prices.
            symbol: Ticker symbol for display.
            timeframe: Candle timeframe (e.g. ``"1h"``).
            account_size: Account equity in USD.
            volume: Optional volume array (reserved for future use).

        Returns:
            :class:`LeverageRecommendation` with all fields populated.
        """
        close = np.asarray(close, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        # --- Feature extraction ---
        features = self.features_engine.compute(close, high, low, volume)
        features = self.features_engine.fill_missing(features)
        feature_array = self.features_engine.to_array(features)

        volatility_ratio = features.get("volatility_ratio", float("nan"))
        vol_ratio_safe = float(volatility_ratio) if not np.isnan(volatility_ratio) else 1.0
        adx = features.get("adx", 25.0)
        adx_norm = min(float(adx) / 100.0, 1.0) if not np.isnan(adx) else 0.25
        current_drawdown = features.get("current_drawdown", 0.0)
        trend_direction = int(features.get("trend_direction", 0))
        hist_vol = features.get("hist_vol_20", 0.3)
        volatility_spike = bool(features.get("volatility_spike", False))

        # --- Tier classification ---
        tier_info = classify_volatility_tier(max(vol_ratio_safe, 1e-9))

        # --- Leverage recommendation ---
        if self._model_loaded:
            try:
                raw_leverage, confidence = self._ml_recommendation(feature_array)
            except Exception:
                raw_leverage, confidence = self._heuristic_recommendation(tier_info, adx, current_drawdown)
        else:
            raw_leverage, confidence = self._heuristic_recommendation(tier_info, adx, current_drawdown)

        # Never exceed tier's max leverage
        tier_cap = tier_info.max_leverage
        recommended_leverage = max(1, min(raw_leverage, tier_cap))

        # Apply risk-based cap
        recommended_leverage = self.calculator.recommended_leverage_from_risk(
            account_size=account_size,
            risk_percent=self.risk_percent,
            stop_loss_percent=self.stop_loss_percent,
            model_recommendation=recommended_leverage,
        )

        # Aggressive = tier max; conservative / max_safe = recommended
        aggressive_leverage = max(1, min(tier_cap, self.calculator.MAX_ABSOLUTE_LEVERAGE))
        max_safe_leverage = recommended_leverage

        # --- Liquidation risk ---
        hist_vol_val = float(hist_vol) if not np.isnan(hist_vol) else 0.3
        liq_risk = self.calculator.liquidation_risk(
            volatility=hist_vol_val,
            trend_strength=adx_norm,
            leverage=recommended_leverage,
        )
        risk_level = self.calculator.risk_level_label(liq_risk)

        # Safety score: inverse of liquidation risk
        safety_score = max(0.0, 100.0 - liq_risk)

        # --- Safety checks ---
        safety_report = self.safety_checker.run_checks(
            account_size=account_size,
            data_length=len(close),
            volatility_ratio=vol_ratio_safe,
            volatility_spike=volatility_spike,
            current_drawdown=float(current_drawdown),
            requested_leverage=recommended_leverage,
            liquidation_risk_pct=liq_risk,
        )

        # If safety checks fail, reduce leverage
        adjustment_needed = False
        adjustment_reason = ""
        if not safety_report.passed:
            recommended_leverage = 1
            adjustment_needed = True
            critical = [i for i in safety_report.issues if not i.passed]
            adjustment_reason = "; ".join(i.message for i in critical[:3])

        # --- Account summary ---
        acct = self.calculator.account_summary(
            account_size=account_size,
            leverage=recommended_leverage,
            risk_percent=self.risk_percent,
            stop_loss_percent=self.stop_loss_percent,
            daily_loss_limit_pct=self.daily_loss_limit_pct,
        )

        return LeverageRecommendation(
            symbol=symbol,
            timeframe=timeframe,
            recommended_leverage=recommended_leverage,
            max_safe_leverage=max_safe_leverage,
            aggressive_leverage=aggressive_leverage,
            confidence_score=round(confidence, 2),
            safety_score=round(safety_score, 2),
            liquidation_risk_pct=round(liq_risk, 2),
            volatility_tier=tier_info.tier,
            volatility_ratio=round(vol_ratio_safe, 4),
            trend_direction=trend_direction,
            trend_strength=round(adx_norm, 4),
            account_size=account_size,
            buying_power=acct["buying_power"],
            max_position_size=acct["max_position_size"],
            daily_stop_loss=acct["daily_stop_loss"],
            risk_level=risk_level,
            safety_report=safety_report,
            adjustment_needed=adjustment_needed,
            adjustment_reason=adjustment_reason,
        )

    def analyze(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> dict:
        """Return a detailed market analysis without account-specific calculations.

        Useful for the ``GET /leverage/analysis/{symbol}`` endpoint.
        """
        close = np.asarray(close, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        features = self.features_engine.compute(close, high, low)
        vol_ratio = features.get("volatility_ratio", float("nan"))
        vol_ratio_safe = float(vol_ratio) if not np.isnan(vol_ratio) else 1.0
        tier_info = classify_volatility_tier(max(vol_ratio_safe, 1e-9))

        return {
            "symbol": symbol,
            "volatility_tier": tier_info.tier.value,
            "volatility_ratio": round(vol_ratio_safe, 4),
            "max_leverage_for_tier": tier_info.max_leverage,
            "tier_description": tier_info.description,
            "liquidation_risk_range": tier_info.liquidation_risk_range,
            "features": {k: (round(v, 6) if not np.isnan(v) else None) for k, v in features.items()},
        }
