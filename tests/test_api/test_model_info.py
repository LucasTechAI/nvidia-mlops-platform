"""
Tests for model info endpoints.
"""

from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint dict."""
    state_dict = {
        "lstm.weight_ih_l0": torch.randn(512, 5),
        "lstm.weight_hh_l0": torch.randn(512, 128),
        "lstm.bias_ih_l0": torch.randn(512),
        "lstm.bias_hh_l0": torch.randn(512),
        "fc.weight": torch.randn(1, 128),
        "fc.bias": torch.randn(1),
    }
    return {
        "model_state_dict": state_dict,
        "model_config": {
            "input_size": 5,
            "hidden_size": 128,
            "output_size": 1,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
        },
        "epoch": 100,
        "best_epoch": 85,
        "loss": 0.0023,
        "best_loss": 0.0019,
        "training_info": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        "test_metrics": {
            "rmse": 3.45,
            "mae": 2.12,
            "r2": 0.91,
            "mape": 1.56,
        },
        "features": ["close", "volume", "high", "low", "open"],
        "training_history": {
            "train_loss": [0.05, 0.03, 0.02, 0.015, 0.012],
            "val_loss": [0.06, 0.04, 0.025, 0.018, 0.015],
            "train_rmse": [5.0, 4.0, 3.5, 3.2, 3.0],
            "val_rmse": [5.5, 4.3, 3.8, 3.5, 3.3],
        },
        "hpo_best_params": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.001,
            "dropout": 0.2,
        },
    }


class TestGetModelInfo:
    """Tests for GET /model/info."""

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_200_with_checkpoint(self, mock_load, client, mock_checkpoint):
        """Test model info returns 200 when checkpoint exists."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/info")
        assert response.status_code == 200

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_response_structure(self, mock_load, client, mock_checkpoint):
        """Test model info response has required fields."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/info")
        data = response.json()

        assert "model_config" in data
        assert "parameters" in data
        assert "training_info" in data
        assert "test_metrics" in data
        assert "epoch" in data
        assert "features" in data

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_model_config_fields(self, mock_load, client, mock_checkpoint):
        """Test model config has architecture fields."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/info")
        config = response.json()["model_config"]

        assert config["input_size"] == 5
        assert config["hidden_size"] == 128
        assert config["output_size"] == 1

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_parameter_counts(self, mock_load, client, mock_checkpoint):
        """Test parameter analysis is present."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/info")
        params = response.json()["parameters"]

        assert "total" in params
        assert "trainable" in params
        assert "layers" in params
        assert params["total"] > 0

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_test_metrics_values(self, mock_load, client, mock_checkpoint):
        """Test that test metrics are returned correctly."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/info")
        metrics = response.json()["test_metrics"]

        assert metrics["rmse"] == pytest.approx(3.45, abs=0.01)
        assert metrics["r2"] == pytest.approx(0.91, abs=0.01)

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_404_when_no_checkpoint(self, mock_load, client):
        """Test 404 when no checkpoint found."""
        mock_load.return_value = None
        response = client.get("/model/info")
        assert response.status_code == 404

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_handles_empty_config(self, mock_load, client):
        """Test handles checkpoint with minimal data."""
        mock_load.return_value = {
            "model_state_dict": {},
            "model_config": {},
            "epoch": 0,
            "loss": 0.0,
        }
        response = client.get("/model/info")
        assert response.status_code == 200

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_handles_tensor_in_config(self, mock_load, client):
        """Test serialization of tensor values in config."""
        mock_load.return_value = {
            "model_state_dict": {},
            "model_config": {"dropout": torch.tensor(0.2)},
            "epoch": 0,
            "loss": 0.0,
            "training_info": {},
            "test_metrics": {"rmse": torch.tensor(3.45)},
        }
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["test_metrics"]["rmse"] == pytest.approx(3.45, abs=0.01)


class TestGetTrainingHistory:
    """Tests for GET /model/training-history."""

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_200_with_history(self, mock_load, client, mock_checkpoint):
        """Test training history returns 200."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/training-history")
        assert response.status_code == 200

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_history_structure(self, mock_load, client, mock_checkpoint):
        """Test training history has expected keys."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/training-history")
        data = response.json()

        assert "train_loss" in data
        assert "val_loss" in data
        assert len(data["train_loss"]) == 5

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_404_no_checkpoint(self, mock_load, client):
        """Test 404 when no checkpoint."""
        mock_load.return_value = None
        response = client.get("/model/training-history")
        assert response.status_code == 404

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_404_no_history(self, mock_load, client):
        """Test 404 when checkpoint has no history."""
        mock_load.return_value = {"model_state_dict": {}, "training_history": {}}
        response = client.get("/model/training-history")
        assert response.status_code == 404

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_handles_numpy_arrays(self, mock_load, client):
        """Test handles numpy arrays in history."""
        import numpy as np

        mock_load.return_value = {
            "training_history": {
                "train_loss": np.array([0.05, 0.03, 0.02]),
                "val_loss": np.array([0.06, 0.04, 0.025]),
            },
        }
        response = client.get("/model/training-history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["train_loss"], list)


class TestGetHPOResults:
    """Tests for GET /model/hpo-results."""

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_200_from_checkpoint(self, mock_load, client, mock_checkpoint):
        """Test HPO results from checkpoint."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/hpo-results")
        assert response.status_code == 200

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_hpo_response_structure(self, mock_load, client, mock_checkpoint):
        """Test HPO response has source and params."""
        mock_load.return_value = mock_checkpoint
        response = client.get("/model/hpo-results")
        data = response.json()

        assert "source" in data
        assert "best_params" in data
        assert data["source"] == "checkpoint"

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_404_no_checkpoint(self, mock_load, client):
        """Test 404 when no checkpoint."""
        mock_load.return_value = None
        response = client.get("/model/hpo-results")
        assert response.status_code == 404

    @patch("src.api.routers.model_info._load_checkpoint")
    def test_returns_404_no_hpo_data(self, mock_load, client):
        """Test 404 when no HPO data anywhere."""
        mock_load.return_value = {"hpo_best_params": {}}
        with patch(
            "src.api.routers.model_info._load_hpo_from_mlflow",
            side_effect=Exception("No HPO"),
        ):
            response = client.get("/model/hpo-results")
            assert response.status_code == 404
