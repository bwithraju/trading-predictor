"""Tests for ML models (feature engineering, training, prediction)."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.models.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.predictor import TradingPredictor, PredictionResult


class TestFeatureEngineer:
    fe = FeatureEngineer()

    def test_get_feature_matrix(self, ohlcv_df):
        X = self.fe.get_feature_matrix(ohlcv_df)
        assert not X.empty
        assert X.isna().sum().sum() == 0  # no NaNs

    def test_create_labels(self, ohlcv_df):
        y = self.fe.create_labels(ohlcv_df)
        assert len(y) == len(ohlcv_df)
        assert set(y.dropna().unique()).issubset({0, 1})

    def test_build_features(self, ohlcv_df):
        df = self.fe.build_features(ohlcv_df.copy())
        assert "rsi_slope" in df.columns
        assert "price_change_1" in df.columns
        assert "bb_width" in df.columns


class TestModelTrainer:
    def test_train_and_predict(self, ohlcv_df, tmp_path):
        trainer = ModelTrainer(model_dir=str(tmp_path))
        metrics = trainer.train(ohlcv_df, "TESTSTOCK")
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert trainer.model is not None

    def test_save_and_load(self, ohlcv_df, tmp_path):
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.train(ohlcv_df, "LOAD_TEST")
        trainer2 = ModelTrainer(model_dir=str(tmp_path))
        loaded = trainer2.load_model("LOAD_TEST")
        assert loaded is True
        assert trainer2.model is not None

    def test_insufficient_data(self, short_ohlcv_df):
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Not enough data"):
            trainer.train(short_ohlcv_df, "SHORT")


class TestTradingPredictor:
    def test_predict_no_model(self, ohlcv_df):
        predictor = TradingPredictor()
        result = predictor.predict(
            df=ohlcv_df, symbol="NOMODEL",
            entry_price=100.0, risk_percent=2.0,
        )
        assert isinstance(result, PredictionResult)
        assert result.symbol == "NOMODEL"
        assert result.prediction in ("UP", "DOWN", "NO_SIGNAL")

    def test_predict_with_trained_model(self, ohlcv_df, tmp_path):
        import os
        os.environ["MODEL_DIR"] = str(tmp_path)
        trainer = ModelTrainer(model_dir=str(tmp_path))
        trainer.train(ohlcv_df, "TRAINED")

        predictor = TradingPredictor(confidence_threshold=0.0, min_indicator_alignment=0)
        predictor._trainers["TRAINED"] = trainer

        result = predictor.predict(
            df=ohlcv_df, symbol="TRAINED",
            entry_price=100.0, risk_percent=2.0,
        )
        assert result.prediction in ("UP", "DOWN", "NO_SIGNAL")
        assert result.confidence >= 0.0
