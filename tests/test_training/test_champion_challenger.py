"""Tests for training/champion_challenger.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.champion_challenger import (
    IMPROVEMENT_THRESHOLD,
    ComparisonResult,
    ModelMetrics,
    compare_models,
    evaluate_model,
    run_champion_challenger,
    _save_result,
)


# ── ModelMetrics ───────────────────────────────────────────────


class TestModelMetrics:
    def test_defaults(self):
        m = ModelMetrics()
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.r2 == 0.0
        assert m.timestamp

    def test_to_dict(self):
        m = ModelMetrics(rmse=1.5, mae=1.0, mape=5.0, r2=0.9, directional_accuracy=52.0)
        d = m.to_dict()
        assert d["rmse"] == 1.5
        assert d["r2"] == 0.9
        assert "timestamp" in d


# ── ComparisonResult ───────────────────────────────────────────


class TestComparisonResult:
    def test_to_dict(self):
        r = ComparisonResult(
            champion_metrics=ModelMetrics(rmse=10.0),
            challenger_metrics=ModelMetrics(rmse=9.0),
            promote=True,
            reason="improved",
            rmse_delta=-1.0,
            rmse_delta_pct=-0.1,
        )
        d = r.to_dict()
        assert d["promote"] is True
        assert d["rmse_delta"] == -1.0
        assert "champion" in d
        assert "challenger" in d


# ── compare_models ─────────────────────────────────────────────


class TestCompareModels:
    def test_promote_when_significantly_better(self):
        champ = ModelMetrics(rmse=10.0)
        challenger = ModelMetrics(rmse=9.0)  # 10% improvement
        result = compare_models(champ, challenger)
        assert result.promote is True
        assert result.rmse_delta < 0

    def test_no_promote_below_threshold(self):
        champ = ModelMetrics(rmse=10.0)
        challenger = ModelMetrics(rmse=9.996)  # 0.04% — below 0.5%
        result = compare_models(champ, challenger)
        assert result.promote is False
        assert "below threshold" in result.reason

    def test_no_promote_when_worse(self):
        champ = ModelMetrics(rmse=10.0)
        challenger = ModelMetrics(rmse=11.0)  # worse
        result = compare_models(champ, challenger)
        assert result.promote is False
        assert "worse" in result.reason

    def test_custom_threshold(self):
        champ = ModelMetrics(rmse=10.0)
        challenger = ModelMetrics(rmse=9.5)  # 5% improvement
        result = compare_models(champ, challenger, threshold=0.10)  # need 10%
        assert result.promote is False

    def test_champion_rmse_zero(self):
        champ = ModelMetrics(rmse=0.0)
        challenger = ModelMetrics(rmse=0.5)
        result = compare_models(champ, challenger)
        assert result.promote is False

    def test_exact_threshold_boundary(self):
        champ = ModelMetrics(rmse=10.0)
        # Exactly at threshold (0.5% = 0.05 improvement)
        challenger = ModelMetrics(rmse=9.95)
        result = compare_models(champ, challenger, threshold=0.005)
        assert result.promote is True

    def test_improvement_threshold_value(self):
        assert IMPROVEMENT_THRESHOLD == 0.005


# ── evaluate_model ─────────────────────────────────────────────


class TestEvaluateModel:
    def _make_loader(self, n=50, n_features=5):
        X = torch.randn(n, 10, n_features)
        y = torch.randn(n, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16)

    def test_returns_model_metrics(self):
        from src.models.lstm_model import NvidiaLSTM

        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, output_size=5)
        loader = self._make_loader()
        metrics = evaluate_model(model, loader, scaler=None)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.rmse >= 0
        assert 0 <= metrics.mape

    def test_with_scaler(self):
        from sklearn.preprocessing import MinMaxScaler

        from src.models.lstm_model import NvidiaLSTM

        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, output_size=5)
        loader = self._make_loader()
        scaler = MinMaxScaler()
        scaler.fit(np.random.rand(20, 5))
        metrics = evaluate_model(model, loader, scaler=scaler, target_idx=0)
        assert isinstance(metrics, ModelMetrics)

    def test_single_feature_scaler(self):
        from sklearn.preprocessing import MinMaxScaler

        from src.models.lstm_model import NvidiaLSTM

        model = NvidiaLSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1)
        X = torch.randn(20, 10, 1)
        y = torch.randn(20, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        scaler = MinMaxScaler()
        scaler.fit(np.random.rand(20, 1))
        metrics = evaluate_model(model, loader, scaler=scaler, target_idx=0)
        assert metrics.rmse >= 0


# ── _save_result ───────────────────────────────────────────────


class TestSaveResult:
    def test_saves_json(self):
        with patch("src.training.champion_challenger.RESULTS_DIR", Path(tempfile.mkdtemp())):
            from src.training.champion_challenger import RESULTS_DIR

            result = {"promoted": False, "reason": "test"}
            _save_result(result)
            saved = RESULTS_DIR / "latest_comparison.json"
            assert saved.exists()
            data = json.loads(saved.read_text())
            assert data["promoted"] is False


# ── run_champion_challenger ────────────────────────────────────


class TestRunChampionChallenger:
    @patch("src.training.champion_challenger._save_result")
    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_no_drift_skips_retraining(self, mock_drift, mock_save):
        mock_drift.return_value = {"drift_detected": False}
        result = run_champion_challenger(retrain_on_drift=True)
        assert result["drift_detected"] is False
        assert result["retrained"] is False
        assert result["reason"] == "No drift detected"
        mock_save.assert_called_once()

    @patch("src.training.champion_challenger._save_result")
    @patch("src.training.train.train_model")
    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_triggers_training(self, mock_drift, mock_train, mock_save):
        mock_drift.return_value = {"drift_detected": True}
        mock_train.return_value = {
            "run_id": "test123",
            "best_val_loss": 0.04,
            "champion_val_loss": 0.05,
        }
        with patch("mlflow.set_experiment"), patch("mlflow.start_run") as mock_run:
            mock_run.return_value.__enter__ = MagicMock()
            mock_run.return_value.__exit__ = MagicMock(return_value=False)
            with patch("mlflow.log_params"), patch("mlflow.log_metrics"), patch("mlflow.set_tag"):
                result = run_champion_challenger(retrain_on_drift=True)

        assert result["drift_detected"] is True
        assert result["retrained"] is True
        assert result["promoted"] is True  # 0.04 vs 0.05 = 20% improvement

    @patch("src.training.champion_challenger._save_result")
    @patch("src.training.train.train_model")
    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_training_failure(self, mock_drift, mock_train, mock_save):
        mock_drift.return_value = {"drift_detected": True}
        mock_train.side_effect = RuntimeError("Training exploded")
        result = run_champion_challenger(retrain_on_drift=True)
        assert result["retrained"] is False
        assert "Training exploded" in result.get("error", "")

    @patch("src.training.champion_challenger._save_result")
    @patch("src.training.train.train_model")
    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_forced_run_without_drift(self, mock_drift, mock_train, mock_save):
        mock_drift.return_value = {"drift_detected": False}
        mock_train.return_value = {
            "run_id": "abc",
            "best_val_loss": 0.05,
            "champion_val_loss": 0.05,
        }
        with patch("mlflow.set_experiment"), patch("mlflow.start_run") as mock_run:
            mock_run.return_value.__enter__ = MagicMock()
            mock_run.return_value.__exit__ = MagicMock(return_value=False)
            with patch("mlflow.log_params"), patch("mlflow.log_metrics"), patch("mlflow.set_tag"):
                result = run_champion_challenger(retrain_on_drift=False)

        assert result["retrained"] is True
        # Same RMSE → no promotion
        assert result["promoted"] is False

    @patch("src.training.champion_challenger._save_result")
    @patch("src.training.train.train_model")
    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_mlflow_failure_handled(self, mock_drift, mock_train, mock_save):
        mock_drift.return_value = {"drift_detected": True}
        mock_train.return_value = {
            "run_id": "x",
            "best_val_loss": 0.03,
            "champion_val_loss": 0.05,
        }
        with patch("mlflow.set_experiment") as mock_exp:
            mock_exp.side_effect = RuntimeError("MLflow down")
            # Should not raise, just warn
            result = run_champion_challenger()

        assert result["retrained"] is True
