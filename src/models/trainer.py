"""Train and persist ML models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import config
from src.models.features import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)

fe = FeatureEngineer()


class ModelTrainer:
    """Train a Random Forest classifier on historical price data."""

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir or config.model.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        horizon: int = 5,
        threshold: float = 0.01,
    ) -> dict:
        """Train a Random Forest on *df* and save the model.

        Returns a metrics dict with accuracy, f1, and report.
        """
        logger.info("Training model for %s (horizon=%d, threshold=%.3f)", symbol, horizon, threshold)
        X = fe.get_feature_matrix(df)
        y = fe.create_labels(df, horizon=horizon, threshold=threshold)
        # Align indices
        common = X.index.intersection(y.dropna().index)
        X = X.loc[common]
        y = y.loc[common]

        if len(X) < 50:
            raise ValueError(f"Not enough data to train ({len(X)} rows after feature extraction)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1 - config.model.TRAIN_TEST_SPLIT,
            random_state=config.model.RANDOM_STATE,
            shuffle=False,  # preserve temporal order
        )

        self.model = RandomForestClassifier(
            n_estimators=config.model.N_ESTIMATORS,
            random_state=config.model.RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.feature_names = list(X.columns)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        self.save_model(symbol)
        metrics = {"accuracy": round(acc, 4), "f1_score": round(f1, 4), "report": report}
        logger.info("Model trained for %s | accuracy=%.4f | f1=%.4f", symbol, acc, f1)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("-", "_").lower()
        path = self.model_dir / f"{safe}_model.pkl"
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info("Model saved to %s", path)
        return path

    def load_model(self, symbol: str) -> bool:
        safe = symbol.replace("/", "_").replace("-", "_").lower()
        path = self.model_dir / f"{safe}_model.pkl"
        if not path.exists():
            logger.warning("No saved model found for %s at %s", symbol, path)
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        logger.info("Model loaded from %s", path)
        return True
