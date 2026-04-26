"""Extended monitoring API tests: all-triggers drift and runs history."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAllTriggers:
    @patch("src.monitoring.drift.detect_all_triggers")
    def test_all_triggers_returns_200(self, mock_detect, client):
        mock_detect.return_value = {
            "data_drift": {"triggered": False, "psi_max": 0.05},
            "model_staleness": {"triggered": False, "days_since_training": 10},
            "ci_breach": {"triggered": False, "breach_pct": 0.0},
            "any_triggered": False,
        }
        response = client.post("/monitoring/drift/all-triggers")
        assert response.status_code == 200

    @patch("src.monitoring.drift.detect_all_triggers")
    def test_all_triggers_response_structure(self, mock_detect, client):
        report = {
            "data_drift": {"triggered": True, "psi_max": 0.35},
            "model_staleness": {"triggered": False, "days_since_training": 5},
            "ci_breach": {"triggered": False},
            "any_triggered": True,
        }
        mock_detect.return_value = report
        response = client.post("/monitoring/drift/all-triggers")
        data = response.json()
        assert "data_drift" in data
        assert "any_triggered" in data

    @patch("src.monitoring.drift.detect_all_triggers")
    def test_all_triggers_serializes_numpy(self, mock_detect, client):
        import numpy as np

        mock_detect.return_value = {
            "data_drift": {"triggered": np.bool_(True), "psi_max": np.float64(0.35)},
            "model_staleness": {"triggered": np.bool_(False)},
            "ci_breach": {"triggered": np.bool_(False)},
            "any_triggered": np.bool_(True),
        }
        response = client.post("/monitoring/drift/all-triggers")
        assert response.status_code == 200
        data = response.json()
        # numpy booleans may be serialized as string "True" by the endpoint
        assert data["any_triggered"] in (True, "True", 1)

    @patch(
        "src.monitoring.drift.detect_all_triggers",
        side_effect=Exception("trigger failure"),
    )
    def test_all_triggers_returns_500_on_error(self, mock_detect, client):
        response = client.post("/monitoring/drift/all-triggers")
        assert response.status_code == 500

    @patch("src.monitoring.drift.detect_all_triggers")
    def test_all_triggers_called_with_save_results_true(self, mock_detect, client):
        mock_detect.return_value = {"any_triggered": False}
        client.post("/monitoring/drift/all-triggers")
        call_kwargs = mock_detect.call_args[1]
        assert call_kwargs.get("save_results") is True


class TestRunsHistory:
    def test_runs_history_returns_200_empty_dir(self, client):
        with patch(
            "src.api.routers.monitoring_api.PROJECT_ROOT",
            Path("/nonexistent_mlops_test_dir"),
        ):
            response = client.get("/monitoring/runs/history")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
        assert data["total"] == 0
        assert data["runs"] == []

    def test_runs_history_with_file_based_run(self, client, tmp_path):
        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "meta.yaml").write_text("name: TestExperiment\n")

        run_dir = exp_dir / "abc123def456"
        run_dir.mkdir()
        (run_dir / "meta.yaml").write_text(
            "run_name: test_run\nstatus: 3\nstart_time: 1700000000000\nend_time: 1700003600000\n"
        )

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "rmse").write_text("1700000000000 3.14 0\n")

        params_dir = run_dir / "params"
        params_dir.mkdir()
        (params_dir / "lr").write_text("0.001")

        with patch("src.api.routers.monitoring_api.PROJECT_ROOT", tmp_path):
            response = client.get("/monitoring/runs/history")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        run = data["runs"][0]
        assert run["run_id"] == "abc123def456"
        assert run["experiment"] == "TestExperiment"
        assert run["status"] == "FINISHED"
        assert run["metrics"]["rmse"] == pytest.approx(3.14, abs=0.001)
        assert run["params"]["lr"] == "0.001"
        assert run["source"] == "file"

    def test_runs_history_multiple_experiments(self, client, tmp_path):
        mlruns = tmp_path / "mlruns"
        for exp_id in ["1", "2"]:
            exp_dir = mlruns / exp_id
            exp_dir.mkdir(parents=True)
            run_dir = exp_dir / f"run_{exp_id}"
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text(
                f"run_name: run_{exp_id}\nstatus: 3\nstart_time: 1700000000000\nend_time: 1700001000000\n"
            )

        with patch("src.api.routers.monitoring_api.PROJECT_ROOT", tmp_path):
            response = client.get("/monitoring/runs/history")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_runs_history_skips_models_subdir(self, client, tmp_path):
        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        exp_dir.mkdir(parents=True)

        # This directory named "models" should be skipped
        models_dir = exp_dir / "models"
        models_dir.mkdir()
        (models_dir / "meta.yaml").write_text("run_name: should_be_skipped\n")

        # A valid run
        run_dir = exp_dir / "valid_run"
        run_dir.mkdir()
        (run_dir / "meta.yaml").write_text(
            "run_name: valid\nstatus: 3\nstart_time: 1700000000000\nend_time: 1700001000000\n"
        )

        with patch("src.api.routers.monitoring_api.PROJECT_ROOT", tmp_path):
            response = client.get("/monitoring/runs/history")

        data = response.json()
        run_ids = [r["run_id"] for r in data["runs"]]
        assert "models" not in run_ids
        assert "valid_run" in run_ids

    def test_runs_history_sorted_by_start_time_desc(self, client, tmp_path):
        mlruns = tmp_path / "mlruns"
        exp_dir = mlruns / "1"
        exp_dir.mkdir(parents=True)

        run_a = exp_dir / "run_older"
        run_a.mkdir()
        (run_a / "meta.yaml").write_text(
            "run_name: older\nstatus: 3\nstart_time: 1600000000000\nend_time: 1600001000000\n"
        )

        run_b = exp_dir / "run_newer"
        run_b.mkdir()
        (run_b / "meta.yaml").write_text(
            "run_name: newer\nstatus: 3\nstart_time: 1700000000000\nend_time: 1700001000000\n"
        )

        with patch("src.api.routers.monitoring_api.PROJECT_ROOT", tmp_path):
            response = client.get("/monitoring/runs/history")

        data = response.json()
        assert data["total"] == 2
        assert data["runs"][0]["run_name"] == "newer"
        assert data["runs"][1]["run_name"] == "older"
