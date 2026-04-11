"""Tests for the LIME explainability module."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.explainability.lime_explainer import (
    _build_predict_fn,
    explain_batch_with_lime,
    explain_with_lime,
    plot_lime_explanation,
    plot_lime_global,
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
    """Small LSTM for fast tests."""
    m = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, output_size=5)
    m.eval()
    return m


@pytest.fixture
def synthetic_data():
    """X: (50, 10, 5), y: (50, 5)."""
    np.random.seed(42)
    X = np.random.randn(50, 10, 5).astype(np.float32)
    y = np.random.randn(50, 5).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Tests — _build_predict_fn
# ---------------------------------------------------------------------------


class TestBuildPredictFn:
    def test_returns_callable(self, model, synthetic_data, device):
        X, _ = synthetic_data
        fn = _build_predict_fn(model, X[0:1], device, output_index=0)
        assert callable(fn)

    def test_output_shape(self, model, synthetic_data, device):
        X, _ = synthetic_data
        fn = _build_predict_fn(model, X[0:1], device, output_index=0)
        flat = X[:5, -1, :]  # (5, 5)
        preds = fn(flat)
        assert preds.shape == (5,)

    def test_deterministic(self, model, synthetic_data, device):
        X, _ = synthetic_data
        fn = _build_predict_fn(model, X[0:1], device, output_index=0)
        flat = X[:3, -1, :]
        p1 = fn(flat)
        p2 = fn(flat)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_output_index(self, model, synthetic_data, device):
        X, _ = synthetic_data
        fn0 = _build_predict_fn(model, X[0:1], device, output_index=0)
        fn3 = _build_predict_fn(model, X[0:1], device, output_index=3)
        flat = X[:3, -1, :]
        p0 = fn0(flat)
        p3 = fn3(flat)
        # Different output indices should give different results (usually)
        assert p0.shape == p3.shape == (3,)


# ---------------------------------------------------------------------------
# Tests — explain_with_lime
# ---------------------------------------------------------------------------


class TestExplainWithLime:
    def test_result_keys(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_with_lime(
            model,
            X,
            sample_index=0,
            num_samples=50,
            device=device,
        )
        assert "method" in result
        assert result["method"] == "lime"
        assert "feature_weights" in result
        assert "predicted_value" in result
        assert "local_r2" in result
        assert "intercept" in result
        assert "feature_names" in result

    def test_feature_weights_keys(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_with_lime(
            model,
            X,
            sample_index=0,
            num_samples=50,
            device=device,
        )
        # Should have weights for the 5 features
        assert len(result["feature_weights"]) > 0
        assert len(result["feature_weights"]) <= 5

    def test_custom_feature_names(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        names = ["A", "B", "C", "D", "E"]
        result = explain_with_lime(
            model,
            X,
            sample_index=0,
            feature_names=names,
            num_samples=50,
            device=device,
        )
        assert result["feature_names"] == names

    def test_saves_json(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        explain_with_lime(model, X, sample_index=0, num_samples=50, device=device)
        json_path = tmp_path / "lime_explanation.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "feature_weights" in data

    def test_different_samples(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        r1 = explain_with_lime(model, X, sample_index=0, num_samples=50, device=device)
        r2 = explain_with_lime(model, X, sample_index=5, num_samples=50, device=device)
        assert r1["sample_index"] == 0
        assert r2["sample_index"] == 5
        # Predicted values should differ for different samples
        assert r1["predicted_value"] != r2["predicted_value"]

    def test_local_r2_bounded(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_with_lime(model, X, sample_index=0, num_samples=100, device=device)
        # R² should be a finite number (can be negative for bad fits)
        assert np.isfinite(result["local_r2"])

    def test_sorted_weights(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_with_lime(model, X, sample_index=0, num_samples=50, device=device)
        sorted_keys = list(result["feature_weights_sorted"].keys())
        sorted_abs = [abs(result["feature_weights_sorted"][k]) for k in sorted_keys]
        # Should be sorted descending by absolute value
        assert sorted_abs == sorted(sorted_abs, reverse=True)


# ---------------------------------------------------------------------------
# Tests — explain_batch_with_lime
# ---------------------------------------------------------------------------


class TestExplainBatchWithLime:
    def test_result_keys(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_batch_with_lime(
            model,
            X,
            n_explain=3,
            num_samples=30,
            device=device,
        )
        assert result["method"] == "lime_batch"
        assert "mean_abs_weights" in result
        assert "std_abs_weights" in result
        assert "global_ranking" in result
        assert "per_sample_explanations" in result

    def test_n_explained_matches(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_batch_with_lime(
            model,
            X,
            n_explain=5,
            num_samples=30,
            device=device,
        )
        assert result["n_explained"] == 5
        assert len(result["per_sample_explanations"]) == 5

    def test_global_ranking_order(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        result = explain_batch_with_lime(
            model,
            X,
            n_explain=3,
            num_samples=30,
            device=device,
        )
        ranking = result["global_ranking"]
        means = result["mean_abs_weights"]
        # Ranking should be descending by mean absolute weight
        vals = [means[k] for k in ranking]
        assert vals == sorted(vals, reverse=True)

    def test_saves_json(self, model, synthetic_data, device, tmp_path, monkeypatch):
        X, _ = synthetic_data
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        explain_batch_with_lime(model, X, n_explain=3, num_samples=30, device=device)
        json_path = tmp_path / "lime_batch_explanation.json"
        assert json_path.exists()

    def test_n_explain_exceeds_dataset(self, model, device, tmp_path, monkeypatch):
        """n_explain > len(X) should be capped."""
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        X = np.random.randn(5, 10, 5).astype(np.float32)
        result = explain_batch_with_lime(
            model,
            X,
            n_explain=100,
            num_samples=30,
            device=device,
        )
        assert result["n_explained"] == 5


# ---------------------------------------------------------------------------
# Tests — plot_lime_explanation
# ---------------------------------------------------------------------------


class TestPlotLimeExplanation:
    def test_saves_png(self, tmp_path):
        results = {
            "sample_index": 0,
            "local_r2": 0.85,
            "feature_weights": {
                "Open": 0.05,
                "High": -0.03,
                "Low": 0.04,
                "Close": 0.08,
                "Volume": -0.01,
            },
        }
        save_path = str(tmp_path / "lime_test.png")
        returned = plot_lime_explanation(results, save_path=save_path)
        assert Path(returned).exists()

    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        results = {
            "sample_index": 0,
            "local_r2": 0.9,
            "feature_weights": {"A": 0.1, "B": -0.2},
        }
        returned = plot_lime_explanation(results)
        assert Path(returned).exists()


# ---------------------------------------------------------------------------
# Tests — plot_lime_global
# ---------------------------------------------------------------------------


class TestPlotLimeGlobal:
    def test_saves_png(self, tmp_path):
        results = {
            "n_explained": 10,
            "mean_abs_weights": {"Open": 0.05, "High": 0.03, "Low": 0.04, "Close": 0.08, "Volume": 0.01},
            "std_abs_weights": {"Open": 0.01, "High": 0.005, "Low": 0.008, "Close": 0.02, "Volume": 0.003},
        }
        save_path = str(tmp_path / "lime_global_test.png")
        returned = plot_lime_global(results, save_path=save_path)
        assert Path(returned).exists()

    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.explainability.lime_explainer.RESULTS_DIR", tmp_path)
        results = {
            "n_explained": 5,
            "mean_abs_weights": {"A": 0.1, "B": 0.2},
            "std_abs_weights": {"A": 0.01, "B": 0.02},
        }
        returned = plot_lime_global(results)
        assert Path(returned).exists()
