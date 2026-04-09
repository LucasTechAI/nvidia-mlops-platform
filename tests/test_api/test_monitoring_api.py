"""
Tests for monitoring API endpoints.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_drift_results():
    """Create mock drift detection results."""
    return {
        "drift_detected": True,
        "features_analyzed": 5,
        "features_drifted": 2,
        "psi_values": {
            "close": 0.35,
            "volume": 0.25,
            "high": 0.08,
            "low": 0.05,
            "open": 0.12,
        },
        "threshold": 0.2,
    }


@pytest.fixture
def mock_champion_challenger_data():
    """Create mock champion-challenger results."""
    return {
        "champion": {
            "rmse": 3.45,
            "mae": 2.12,
            "r2": 0.91,
            "model_path": "models/champion.pth",
        },
        "challenger": {
            "rmse": 3.20,
            "mae": 1.98,
            "r2": 0.93,
            "model_path": "models/challenger.pth",
        },
        "promoted": True,
        "promotion_reason": "Challenger RMSE improved by 7.2%",
        "timestamp": "2026-04-08T12:00:00",
    }


class TestDriftDetection:
    """Tests for POST /monitoring/drift."""

    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_returns_200(self, mock_detect, client, mock_drift_results):
        """Test drift endpoint returns 200 on success."""
        mock_detect.return_value = mock_drift_results
        response = client.post("/monitoring/drift")
        assert response.status_code == 200

    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_response_structure(self, mock_detect, client, mock_drift_results):
        """Test drift response has expected keys."""
        mock_detect.return_value = mock_drift_results
        response = client.post("/monitoring/drift")
        data = response.json()

        assert "drift_detected" in data
        assert "psi_values" in data
        assert "features_analyzed" in data

    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_psi_values(self, mock_detect, client, mock_drift_results):
        """Test PSI values are returned correctly."""
        mock_detect.return_value = mock_drift_results
        response = client.post("/monitoring/drift")
        data = response.json()

        psi = data["psi_values"]
        assert psi["close"] == pytest.approx(0.35, abs=0.01)
        assert psi["volume"] == pytest.approx(0.25, abs=0.01)

    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_returns_404_no_data(self, mock_detect, client):
        """Test 404 when no data for drift detection."""
        mock_detect.return_value = None
        response = client.post("/monitoring/drift")
        assert response.status_code == 404

    @patch(
        "src.monitoring.drift.detect_drift_from_db",
        side_effect=Exception("DB error"),
    )
    def test_drift_returns_500_on_error(self, mock_detect, client):
        """Test 500 on drift detection failure."""
        response = client.post("/monitoring/drift")
        assert response.status_code == 500

    @patch("src.monitoring.drift.detect_drift_from_db")
    def test_drift_serializes_numpy_values(self, mock_detect, client):
        """Test numpy/tensor values are properly serialized."""
        import numpy as np

        mock_detect.return_value = {
            "drift_detected": np.bool_(True),
            "features_analyzed": np.int64(5),
            "psi_values": {"close": np.float64(0.35), "volume": np.float64(0.25)},
        }
        response = client.post("/monitoring/drift")
        assert response.status_code == 200


class TestChampionChallenger:
    """Tests for GET /monitoring/champion-challenger."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"champion": {"rmse": 3.45}, "challenger": {"rmse": 3.20}}',
    )
    def test_returns_200(self, mock_file, mock_exists, client):
        """Test champion-challenger returns 200."""
        response = client.get("/monitoring/champion-challenger")
        assert response.status_code == 200

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"champion": {"rmse": 3.45}, "challenger": {"rmse": 3.20}, "promoted": true}',
    )
    def test_response_structure(self, mock_file, mock_exists, client):
        """Test response has champion and challenger data."""
        response = client.get("/monitoring/champion-challenger")
        data = response.json()

        assert "champion" in data
        assert "challenger" in data

    def test_returns_404_no_results_file(self, client):
        """Test 404 when results file doesn't exist."""
        with patch(
            "src.api.routers.monitoring_api.PROJECT_ROOT",
            Path("/nonexistent"),
        ):
            response = client.get("/monitoring/champion-challenger")
            assert response.status_code == 404


class TestRunChampionChallenger:
    """Tests for POST /monitoring/champion-challenger/run."""

    @patch("src.training.champion_challenger.run_champion_challenger")
    def test_returns_200_on_success(self, mock_run, client, mock_champion_challenger_data):
        """Test running pipeline returns 200."""
        mock_run.return_value = mock_champion_challenger_data
        response = client.post("/monitoring/champion-challenger/run")
        assert response.status_code == 200

    @patch("src.training.champion_challenger.run_champion_challenger")
    def test_returns_500_on_none_result(self, mock_run, client):
        """Test 500 when pipeline returns None."""
        mock_run.return_value = None
        response = client.post("/monitoring/champion-challenger/run")
        assert response.status_code == 500

    @patch(
        "src.training.champion_challenger.run_champion_challenger",
        side_effect=Exception("Pipeline error"),
    )
    def test_returns_500_on_error(self, mock_run, client):
        """Test 500 on pipeline failure."""
        response = client.post("/monitoring/champion-challenger/run")
        assert response.status_code == 500
