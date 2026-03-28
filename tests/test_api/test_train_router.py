"""Tests for api/routers/train.py endpoints."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import ModelState, get_model_state
from src.api.routers.train import router


def _make_app(state_mock):
    """Create a FastAPI app with overridden dependency."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_state] = lambda: state_mock
    return app


def _make_state(is_training=False):
    state = MagicMock(spec=ModelState)
    state.is_training = is_training
    state.current_epoch = 0
    state.total_epochs = 0
    state.current_loss = None
    state.training_run_id = None
    return state


# ── POST /train ────────────────────────────────────────────────


class TestStartTraining:
    def test_start_training_accepted(self):
        state = _make_state()
        client = TestClient(_make_app(state))
        resp = client.post("/train", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_start_training_conflict(self):
        state = _make_state(is_training=True)
        client = TestClient(_make_app(state))
        resp = client.post("/train", json={})
        assert resp.status_code == 409

    def test_start_training_custom_params(self):
        state = _make_state()
        client = TestClient(_make_app(state))
        resp = client.post(
            "/train",
            json={
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.01,
                "experiment_name": "test_run",
            },
        )
        assert resp.status_code == 200


# ── GET /train/status ──────────────────────────────────────────


class TestTrainingStatus:
    def test_status_idle(self):
        state = _make_state()
        client = TestClient(_make_app(state))
        resp = client.get("/train/status")
        assert resp.status_code == 200
        assert resp.json()["is_training"] is False

    def test_status_in_progress(self):
        state = _make_state(is_training=True)
        state.current_epoch = 3
        state.total_epochs = 10
        state.current_loss = 0.05
        state.training_run_id = "abc123"
        client = TestClient(_make_app(state))
        resp = client.get("/train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_training"] is True
        assert data["current_epoch"] == 3


# ── POST /train/sync ──────────────────────────────────────────


class TestTrainSync:
    def test_sync_conflict(self):
        state = _make_state(is_training=True)
        client = TestClient(_make_app(state))
        resp = client.post("/train/sync", json={})
        assert resp.status_code == 409

    @patch("src.training.train.train_model")
    def test_sync_success(self, mock_train):
        mock_train.return_value = {
            "run_id": "run123",
            "best_val_loss": 0.01,
            "test_metrics": {"rmse": 1.5},
        }
        state = _make_state()
        state.load_model = MagicMock()
        client = TestClient(_make_app(state))
        with patch("src.config.settings") as ms:
            ms.epochs = 5
            ms.batch_size = 32
            ms.learning_rate = 0.001
            ms.hidden_size = 128
            ms.num_layers = 2
            ms.sequence_length = 60
            ms.model_dir = MagicMock()
            # model_dir / "checkpoints" / "best_model.pt"
            ckpt_mock = MagicMock()
            ckpt_mock.exists.return_value = False
            ms.model_dir.__truediv__ = MagicMock(return_value=MagicMock(__truediv__=MagicMock(return_value=ckpt_mock)))
            resp = client.post("/train/sync", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    @patch("src.training.train.train_model", side_effect=RuntimeError("boom"))
    def test_sync_failure(self, mock_train):
        state = _make_state()
        client = TestClient(_make_app(state))
        with patch("src.config.settings") as ms:
            ms.epochs = 1
            ms.batch_size = 32
            ms.learning_rate = 0.001
            ms.hidden_size = 128
            ms.num_layers = 2
            ms.sequence_length = 60
            resp = client.post("/train/sync", json={})
        assert resp.status_code == 500


# ── POST /train/stop ───────────────────────────────────────────


class TestStopTraining:
    def test_stop_no_training(self):
        state = _make_state()
        client = TestClient(_make_app(state))
        resp = client.post("/train/stop")
        assert resp.status_code == 200
        assert "No training" in resp.json()["message"]

    def test_stop_during_training(self):
        state = _make_state(is_training=True)
        client = TestClient(_make_app(state))
        resp = client.post("/train/stop")
        assert resp.status_code == 200
        assert "Stop requested" in resp.json()["message"]
