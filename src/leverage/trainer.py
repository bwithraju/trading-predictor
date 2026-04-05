"""Model training pipeline for the leverage prediction model.

:class:`LeverageTrainer` generates synthetic training data from OHLCV
price series, labels each bar with the correct volatility tier and a
binary "safe" label, trains a :class:`~sklearn.ensemble.RandomForestClassifier`
and a :class:`~sklearn.ensemble.GradientBoostingRegressor`, and persists
the artefacts alongside a :class:`~sklearn.preprocessing.StandardScaler`.

Training can also accept real historical data from any source – just pass
a list of (close, high, low) arrays.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import LeverageFeatures
from .model import LeverageModel
from .tiers import classify_volatility_tier

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    import joblib

    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JOBLIB_AVAILABLE = False

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class LeverageTrainer:
    """Train the leverage prediction model on historical OHLCV data.

    Example::

        trainer = LeverageTrainer()
        # Build training data from multiple assets
        X, y_cls, y_reg = trainer.build_training_data(
            price_arrays=[(close_btc, high_btc, low_btc), ...]
        )
        model = trainer.train(X, y_cls, y_reg)
        model.save_model()
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        window: int = 150,
        step: int = 1,
    ) -> None:
        """
        Args:
            model_dir: Where to save artefacts.
            window: Number of bars in each training window.
            step: Step size when sliding the window (1 = every bar).
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.window = window
        self.step = step
        self.features_engine = LeverageFeatures()

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        n_bars: int = 2000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic OHLCV price data for testing/bootstrapping.

        Args:
            n_bars: Number of bars to generate.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (close, high, low) arrays.
        """
        rng = np.random.default_rng(seed)
        log_returns = rng.normal(loc=0.0002, scale=0.02, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        noise = np.abs(rng.normal(loc=0, scale=0.005, size=n_bars)) * close
        high = close + noise
        low = close - noise
        return close, high, low

    # ------------------------------------------------------------------
    # Feature matrix builder
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        price_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix and labels from a list of price arrays.

        Args:
            price_arrays: List of (close, high, low) tuples.

        Returns:
            (X, y_cls, y_reg) where:
              - X is shape (n_samples, n_features)
              - y_cls is binary safe/risky (1 = safe, 0 = risky)
              - y_reg is recommended leverage integer
        """
        all_X: list[np.ndarray] = []
        all_y_cls: list[int] = []
        all_y_reg: list[int] = []

        for close, high, low in price_arrays:
            close = np.asarray(close, dtype=float)
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)

            end = len(close)
            for start in range(0, end - self.window, self.step):
                seg_close = close[start: start + self.window]
                seg_high = high[start: start + self.window]
                seg_low = low[start: start + self.window]

                feat = self.features_engine.compute(seg_close, seg_high, seg_low)
                feat = self.features_engine.fill_missing(feat)
                x = self.features_engine.to_array(feat)

                vol_ratio = feat.get("volatility_ratio", 1.0)
                vol_ratio_safe = float(vol_ratio) if not np.isnan(vol_ratio) else 1.0

                tier_info = classify_volatility_tier(max(vol_ratio_safe, 1e-9))
                lev = tier_info.max_leverage
                safe = 1 if lev >= 4 else 0  # 4× and above → "safe" class

                all_X.append(x)
                all_y_cls.append(safe)
                all_y_reg.append(lev)

        if not all_X:
            raise ValueError("No training samples generated – check input data length vs window size")

        return (
            np.array(all_X, dtype=float),
            np.array(all_y_cls, dtype=int),
            np.array(all_y_reg, dtype=int),
        )

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        y_reg: np.ndarray,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> LeverageModel:
        """Train the ensemble and return a :class:`LeverageModel`.

        Args:
            X: Feature matrix (n_samples, n_features).
            y_cls: Binary classification labels (safe=1 / risky=0).
            y_reg: Regression labels (recommended leverage 1–15).
            n_estimators: Number of trees/estimators.
            random_state: Seed for reproducibility.

        Returns:
            A :class:`LeverageModel` with trained internals ready to use.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for training. Install with: pip install scikit-learn"
            )  # pragma: no cover

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Class weights for imbalanced data
        classes = np.unique(y_cls)
        if len(classes) > 1:
            weights = compute_class_weight("balanced", classes=classes, y=y_cls)
            class_weight = dict(zip(classes.tolist(), weights.tolist()))
        else:
            class_weight = None

        # Walk-forward cross-validation (just for logging – not used in final fit)
        tscv = TimeSeriesSplit(n_splits=5)
        clf_scores: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_idx, val_idx in tscv.split(X_scaled):
                tmp_clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight=class_weight,
                    n_jobs=-1,
                )
                tmp_clf.fit(X_scaled[train_idx], y_cls[train_idx])
                clf_scores.append(float(tmp_clf.score(X_scaled[val_idx], y_cls[val_idx])))

        # Final models trained on all data
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        clf.fit(X_scaled, y_cls)

        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=4,
        )
        gbr.fit(X_scaled, y_reg)

        # Bundle into LeverageModel
        lev_model = LeverageModel(model_dir=self.model_dir)
        lev_model._clf = clf
        lev_model._gbr = gbr
        lev_model._scaler = scaler
        lev_model._model_loaded = True

        return lev_model

    def train_from_synthetic(
        self,
        n_bars: int = 2000,
        seed: int = 42,
        n_estimators: int = 100,
    ) -> LeverageModel:
        """Convenience method: generate synthetic data, train, and return the model.

        Useful for smoke tests and bootstrapping when real data is unavailable.
        """
        close, high, low = self.generate_synthetic_data(n_bars=n_bars, seed=seed)
        X, y_cls, y_reg = self.build_training_data([(close, high, low)])
        return self.train(X, y_cls, y_reg, n_estimators=n_estimators, random_state=seed)
