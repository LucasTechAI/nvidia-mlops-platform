"""
Tests for evaluation API endpoints.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_comparison_data():
    """Create mock latest_comparison.json content."""
    return {
        "champion": {
            "rmse": 3.45,
            "mae": 2.12,
            "r2": 0.91,
        },
        "challenger": {
            "rmse": 3.20,
            "mae": 1.98,
            "r2": 0.93,
        },
        "promoted": True,
        "promotion_reason": "Challenger outperformed champion on RMSE",
        "timestamp": "2026-04-08T12:00:00",
    }


@pytest.fixture
def mock_golden_set():
    """Create mock golden set data."""
    return [
        {
            "question": "What is NVIDIA stock price?",
            "expected": "The current NVIDIA stock price is $450",
            "category": "price",
        },
        {
            "question": "What was the highest price this year?",
            "expected": "The highest NVIDIA stock price this year was $500",
            "category": "historical",
        },
    ]


class TestGetEvaluationResults:
    """Tests for GET /evaluation/results."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_200(self, mock_file, mock_exists, client, mock_comparison_data):
        """Test evaluation results returns 200."""
        mock_file.return_value.read.return_value = json.dumps(mock_comparison_data)
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)

        with patch("json.load", return_value=mock_comparison_data):
            response = client.get("/evaluation/results")
            assert response.status_code == 200

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_response_structure(self, mock_file, mock_exists, client, mock_comparison_data):
        """Test response has expected fields."""
        with patch("json.load", return_value=mock_comparison_data):
            response = client.get("/evaluation/results")
            data = response.json()

            assert "champion" in data
            assert "challenger" in data
            assert "promoted" in data
            assert "promotion_reason" in data

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_champion_metrics(self, mock_file, mock_exists, client, mock_comparison_data):
        """Test champion metrics are returned correctly."""
        with patch("json.load", return_value=mock_comparison_data):
            response = client.get("/evaluation/results")
            champion = response.json()["champion"]

            assert champion["rmse"] == pytest.approx(3.45, abs=0.01)
            assert champion["r2"] == pytest.approx(0.91, abs=0.01)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_promotion_flag(self, mock_file, mock_exists, client, mock_comparison_data):
        """Test promotion status is returned."""
        with patch("json.load", return_value=mock_comparison_data):
            response = client.get("/evaluation/results")
            data = response.json()

            assert data["promoted"] is True
            assert "outperformed" in data["promotion_reason"]

    def test_returns_404_no_file(self, client):
        """Test 404 when results file doesn't exist."""
        with patch(
            "src.api.routers.evaluation_api.PROJECT_ROOT",
            Path("/nonexistent"),
        ):
            response = client.get("/evaluation/results")
            assert response.status_code == 404


class TestRunExplainability:
    """Tests for POST /evaluation/explainability."""

    @patch("src.api.dependencies.model_state")
    @patch("src.explainability.feature_importance.compute_permutation_importance")
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_200_with_list_results(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test explainability returns 200 with list format results."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        mock_compute.return_value = {
            "feature_names": ["close", "volume", "high"],
            "importances_mean": [0.45, 0.32, 0.15],
            "importances_std": [0.01, 0.02, 0.01],
        }
        response = client.post("/evaluation/explainability")
        assert response.status_code == 200

    @patch("src.api.dependencies.model_state")
    @patch("src.explainability.feature_importance.compute_permutation_importance")
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_features_list(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test response contains features list."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        mock_compute.return_value = {
            "feature_names": ["close"],
            "importances_mean": [0.45],
            "importances_std": [0.01],
        }
        response = client.post("/evaluation/explainability")
        data = response.json()

        assert "features" in data
        assert len(data["features"]) == 1
        assert data["features"][0]["feature"] == "close"

    @patch("src.api.dependencies.model_state")
    @patch("src.explainability.feature_importance.compute_permutation_importance")
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_handles_dict_results(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test handles dict format results."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        mock_compute.return_value = {
            "feature_names": ["close", "volume"],
            "importances_mean": [0.45, 0.32],
            "importances_std": [0.01, 0.02],
        }
        response = client.post("/evaluation/explainability")
        assert response.status_code == 200

        data = response.json()
        assert len(data["features"]) == 2

    @patch("src.api.dependencies.model_state")
    @patch("src.explainability.feature_importance.compute_permutation_importance")
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_404_no_results(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test 404 when no results."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        mock_compute.return_value = None
        response = client.post("/evaluation/explainability")
        assert response.status_code == 404

    @patch("src.api.dependencies.model_state")
    @patch(
        "src.explainability.feature_importance.compute_permutation_importance",
        side_effect=Exception("Computation failed"),
    )
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_500_on_error(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test 500 on computation failure."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        response = client.post("/evaluation/explainability")
        assert response.status_code == 500

    @patch("src.api.dependencies.model_state")
    @patch("src.explainability.feature_importance.compute_permutation_importance")
    @patch("numpy.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_serializes_numpy_values(self, mock_exists, mock_np_load, mock_compute, mock_ms, client):
        """Test numpy values are properly serialized."""
        import numpy as np

        mock_ms.model = MagicMock()
        mock_np_load.return_value = np.zeros((10, 5, 1))
        mock_compute.return_value = {
            "feature_names": ["close", "volume"],
            "importances_mean": [np.float64(0.45), np.float64(0.32)],
            "importances_std": [np.float64(0.01), np.float64(0.02)],
        }
        response = client.post("/evaluation/explainability")
        assert response.status_code == 200

    def test_returns_503_no_model(self, client):
        """Test 503 when model is not loaded."""
        with patch("src.api.dependencies.model_state") as mock_ms:
            mock_ms.model = None
            response = client.post("/evaluation/explainability")
            assert response.status_code == 503


class TestGetLLMResults:
    """Tests for GET /evaluation/llm-results."""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob", return_value=[])
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_200_with_golden_set(self, mock_file, mock_glob, mock_exists, client, mock_golden_set):
        """Test LLM results returns 200."""
        mock_exists.return_value = True
        with patch("json.load", return_value=mock_golden_set):
            response = client.get("/evaluation/llm-results")
            assert response.status_code == 200

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob", return_value=[])
    @patch("builtins.open", new_callable=mock_open)
    def test_response_has_golden_set(self, mock_file, mock_glob, mock_exists, client, mock_golden_set):
        """Test response contains golden set."""
        mock_exists.return_value = True
        with patch("json.load", return_value=mock_golden_set):
            response = client.get("/evaluation/llm-results")
            data = response.json()

            assert "golden_set" in data
            assert "ragas" in data
            assert "llm_judge" in data
            assert len(data["golden_set"]) == 2

    def test_returns_404_no_golden_set(self, client):
        """Test 404 when golden set file missing."""
        with patch(
            "src.api.routers.evaluation_api.PROJECT_ROOT",
            Path("/nonexistent"),
        ):
            response = client.get("/evaluation/llm-results")
            assert response.status_code == 404
