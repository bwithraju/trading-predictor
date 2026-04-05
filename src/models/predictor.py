"""Main prediction pipeline with safety thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import config
from src.analysis.signals import SignalGenerator
from src.models.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.risk.calculator import RiskCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)

signal_gen = SignalGenerator()
fe = FeatureEngineer()


@dataclass
class PredictionResult:
    symbol: str
    prediction: str          # 'UP', 'DOWN', 'NO_SIGNAL'
    confidence: float
    direction: str           # 'LONG', 'SHORT', 'NONE'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_loss: Optional[float] = None
    position_size: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    entry_price: Optional[float] = None
    indicators: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    reason: str = ""


class TradingPredictor:
    """Run the full prediction pipeline: indicators → signals → ML → risk."""

    def __init__(
        self,
        confidence_threshold: float = None,
        min_indicator_alignment: int = None,
    ):
        self.confidence_threshold = confidence_threshold or config.model.CONFIDENCE_THRESHOLD
        self.min_indicator_alignment = min_indicator_alignment or config.model.MIN_INDICATOR_ALIGNMENT
        self._trainers: dict[str, ModelTrainer] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trainer(self, symbol: str) -> ModelTrainer:
        if symbol not in self._trainers:
            t = ModelTrainer()
            t.load_model(symbol)
            self._trainers[symbol] = t
        return self._trainers[symbol]

    def _ml_prediction(
        self, df: pd.DataFrame, symbol: str
    ) -> tuple[str, float]:
        """Return (ml_direction, confidence) from the trained model, or ('NONE', 0)."""
        trainer = self._get_trainer(symbol)
        if trainer.model is None:
            return "NONE", 0.0

        X = fe.get_feature_matrix(df)
        if X.empty:
            return "NONE", 0.0

        # Align to training features
        available = [f for f in trainer.feature_names if f in X.columns]
        if not available:
            return "NONE", 0.0
        X = X[available].iloc[[-1]]

        proba = trainer.model.predict_proba(X)[0]
        classes = list(trainer.model.classes_)

        # Class 1 = UP, Class 0 = DOWN
        up_conf = proba[classes.index(1)] if 1 in classes else 0.0
        down_conf = proba[classes.index(0)] if 0 in classes else 0.0

        if up_conf >= down_conf:
            return "UP", float(up_conf)
        return "DOWN", float(down_conf)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        entry_price: float,
        risk_percent: float = None,
        account_size: float = 10_000.0,
    ) -> PredictionResult:
        """Run the full prediction pipeline and return a :class:`PredictionResult`."""
        signals = signal_gen.generate_from_df(df)
        ml_dir, confidence = self._ml_prediction(df, symbol)

        indicators = {}
        if not df.empty:
            last = df.iloc[-1].to_dict()
            indicators = {k: (None if pd.isna(v) else float(v)) for k, v in last.items()}

        # ---- Safety threshold 1: ML confidence ----
        if confidence < self.confidence_threshold or ml_dir == "NONE":
            reason = (
                f"ML confidence {confidence:.1%} below threshold "
                f"{self.confidence_threshold:.1%}"
            )
            return PredictionResult(
                symbol=symbol,
                prediction="NO_SIGNAL",
                confidence=confidence,
                direction="NONE",
                entry_price=entry_price,
                indicators=indicators,
                signals=signals,
                reason=reason,
            )

        # ---- Safety threshold 2: indicator alignment ----
        if ml_dir == "UP":
            aligned = signals.get("bull_count", 0)
        else:
            aligned = signals.get("bear_count", 0)

        if aligned < self.min_indicator_alignment:
            reason = (
                f"Only {aligned}/{5} indicators confirm {ml_dir} "
                f"(need {self.min_indicator_alignment})"
            )
            return PredictionResult(
                symbol=symbol,
                prediction="NO_SIGNAL",
                confidence=confidence,
                direction="NONE",
                entry_price=entry_price,
                indicators=indicators,
                signals=signals,
                reason=reason,
            )

        # ---- Both thresholds passed: generate trade signal ----
        direction = "LONG" if ml_dir == "UP" else "SHORT"
        risk_calc = RiskCalculator()
        risk = risk_calc.calculate(
            entry_price=entry_price,
            direction=direction,
            risk_percent=risk_percent or config.risk.DEFAULT_RISK_PERCENT,
            account_size=account_size,
        )

        reason = (
            f"{ml_dir} with {confidence:.1%} confidence; "
            f"{aligned}/5 indicators aligned; "
            f"SL={risk.stop_loss:.4f}, TP={risk.take_profit:.4f}"
        )

        return PredictionResult(
            symbol=symbol,
            prediction=ml_dir,
            confidence=confidence,
            direction=direction,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
            max_loss=risk.max_loss,
            position_size=risk.position_size,
            risk_reward_ratio=risk.risk_reward_ratio,
            entry_price=entry_price,
            indicators=indicators,
            signals=signals,
            reason=reason,
        )
