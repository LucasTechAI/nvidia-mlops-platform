"""Tests for the feature importance explainability module."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.explainability.feature_importance import (
    _compute_rmse,
    compute_permutation_importance,
    plot_feature_importance,
)
from src.models.lstm_model import NvidiaLSTM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    return NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, output_size=5)


@pytest.fixture
def synthetic_data():
    """X: (50, 10, 5), y: (50, 5)."""
    np.random.seed(42)
    X = np.random.randn(50, 10, 5).astype(np.float32)
    y = np.random.randn(50, 5).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Tests — _compute_rmse
# ---------------------------------------------------------------------------

class TestComputeRmse:
    def test_returns_positive_float(self, model, synthetic_data, device):
        X, y = synthetic_data
        rmse = _compute_rmse(model, X, y, device)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_perfect_prediction_rmse_near_zero(self, device):
        """If model output == y, RMSE should be ~0."""
        model = NvidiaLSTM(input_size=1, hidden_size=4, num_layers=1, dropout=0.0, output_size=1)
        X = np.zeros((10, 5, 1), dtype=np.float32)
        with torch.no_grad():
            y = model(torch.FloatTensor(X)).numpy()
        rmse = _compute_rmse(model, X, y, device)
        assert rmse < 1e-6


# ---------------------------------------------------------------------------
# Tests — compute_permutation_importance
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    def test_result_keys(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, y = synthetic_data
        # Redirect output directory to tmp
        monkeypatch.setattr(
            "src.explainability.feature_importance.RESULTS_DIR", tmp_path
        )
        result = compute_permutation_importance(
            model, X, y, n_repeats=2, device=device
        )
        assert "feature_names" in result
        assert "importances_mean" in result
        assert "importances_std" in result
        assert "baseline_rmse" in result
        assert "details" in result

    def test_feature_count_matches(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, y = synthetic_data
        monkeypatch.setattr(
            "src.explainability.feature_importance.RESULTS_DIR", tmp_path
        )
        result = compute_permutation_importance(
            model, X, y, n_repeats=2, device=device
        )
        assert len(result["feature_names"]) == 5
        assert len(result["importances_mean"]) == 5
        assert len(result["importances_std"]) == 5

    def test_custom_feature_names(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, y = synthetic_data
        monkeypatch.setattr(
            "src.explainability.feature_importance.RESULTS_DIR", tmp_path
        )
        names = ["A", "B", "C", "D", "E"]
        result = compute_permutation_importance(
            model, X, y, feature_names=names, n_repeats=2, device=device
        )
        assert result["feature_names"] == names

    def test_saves_json(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, y = synthetic_data
        monkeypatch.setattr(
            "src.explainability.feature_importance.RESULTS_DIR", tmp_path
        )
        compute_permutation_importance(model, X, y, n_repeats=2, device=device)
        json_path = tmp_path / "permutation_importance.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "importances_mean" in data


# ---------------------------------------------------------------------------
# Tests — plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_saves_png(self, tmp_path):
        results = {
            "feature_names": ["Open", "High", "Low", "Close", "Volume"],
            "importances_mean": [0.05, 0.03, 0.04, 0.08, 0.01],
            "importances_std": [0.01, 0.005, 0.008, 0.02, 0.003],
            "baseline_rmse": 0.5,
        }
        save_path = str(tmp_path / "importance.png")
        returned_path = plot_feature_importance(results, save_path=save_path)
        assert Path(returned_path).exists()

    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.explainability.feature_importance.RESULTS_DIR", tmp_path
        )
        results = {
            "feature_names": ["A", "B"],
            "importances_mean": [0.1, 0.2],
            "importances_std": [0.01, 0.02],
            "baseline_rmse": 0.3,
        }
        returned_path = plot_feature_importance(results)
        assert Path(returned_path).exists()
