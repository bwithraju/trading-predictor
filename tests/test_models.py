"""Unit tests for src/models/."""
import numpy as np
import pandas as pd
import pytest

from src.models.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.predictor import TradingPredictor


def _make_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestFeatureEngineer:
    fe = FeatureEngineer()

    def test_build_features_adds_derived_columns(self):
        df = _make_df(100)
        enriched = self.fe.build_features(df)
        assert "close_vs_sma20" in enriched.columns
        assert "bb_width" in enriched.columns
        assert "price_change_1" in enriched.columns

    def test_get_feature_matrix_no_nans(self):
        df = _make_df(100)
        X = self.fe.get_feature_matrix(df)
        assert not X.isnull().any().any()

    def test_create_labels_binary(self):
        df = _make_df(100)
        labels = self.fe.create_labels(df)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1})


class TestModelTrainer:
    def test_train_returns_metrics(self, tmp_path):
        df = _make_df(200)
        trainer = ModelTrainer(model_dir=str(tmp_path))
        metrics = trainer.train(df, symbol="TEST")
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_and_load_model(self, tmp_path):
        df = _make_df(200)
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.train(df, symbol="TEST")
        trainer2 = ModelTrainer(model_dir=str(tmp_path))
        loaded = trainer2.load_model("TEST")
        assert loaded is True
        assert trainer2.model is not None

    def test_train_insufficient_data_raises(self, tmp_path):
        df = _make_df(30)
        trainer = ModelTrainer(model_dir=str(tmp_path))
        with pytest.raises(ValueError):
            trainer.train(df, symbol="SHORT")


class TestTradingPredictor:
    def test_predict_no_signal_without_model(self):
        df = _make_df(200)
        predictor = TradingPredictor()
        result = predictor.predict(df=df, symbol="NOMODEL", entry_price=100.0)
        # Without a trained model, we expect NO_SIGNAL
        assert result.prediction == "NO_SIGNAL"

    def test_predict_with_trained_model(self, tmp_path):
        df = _make_df(300)
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.train(df, symbol="SYM")

        predictor = TradingPredictor(confidence_threshold=0.0, min_indicator_alignment=0)
        predictor._trainers["SYM"] = trainer
        result = predictor.predict(df=df, symbol="SYM", entry_price=100.0)
        assert result.prediction in ("UP", "DOWN", "NO_SIGNAL")
        assert result.confidence >= 0.0
