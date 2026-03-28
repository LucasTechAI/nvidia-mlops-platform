"""Tests for api/routers/predict.py — predict & inference endpoints."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from src.api.dependencies import ModelState, get_model_state
from src.api.routers.predict import (
    generate_forecast_with_uncertainty,
    get_forecast_dates,
    router,
)
from src.models.lstm_model import NvidiaLSTM


# ── Helper: forecast dates ─────────────────────────────────────


class TestGetForecastDates:
    def test_skips_weekends(self):
        # Friday
        friday = pd.Timestamp("2024-01-05")
        dates = get_forecast_dates(friday, 3)
        assert len(dates) == 3
        # Mon, Tue, Wed
        assert dates[0].weekday() == 0  # Monday
        assert dates[1].weekday() == 1
        assert dates[2].weekday() == 2

    def test_correct_count(self):
        dates = get_forecast_dates(pd.Timestamp("2024-01-01"), 10)
        assert len(dates) == 10
        for d in dates:
            assert d.weekday() < 5  # all business days


# ── Helper: MC Dropout forecast ────────────────────────────────


class TestGenerateForecastWithUncertainty:
    def test_returns_mean_and_std(self):
        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, output_size=5, dropout=0.1)
        model.train()
        seq = torch.randn(1, 10, 5)
        mean, std = generate_forecast_with_uncertainty(model, seq, horizon=3, n_samples=5, device="cpu")
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std >= 0)

    def test_model_back_to_eval(self):
        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, output_size=5, dropout=0.1)
        seq = torch.randn(1, 10, 5)
        generate_forecast_with_uncertainty(model, seq, horizon=2, n_samples=2, device="cpu")
        assert not model.training  # should be eval


# ── Endpoint tests ─────────────────────────────────────────────


def _make_mock_state(ready=True):
    """Build a mock ModelState with a real small model + scaler."""
    state = MagicMock(spec=ModelState)
    state.is_ready = ready

    model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, output_size=5)
    model.eval()
    state.model = model
    state.device = "cpu"

    scaler = MinMaxScaler()
    scaler.fit(np.random.rand(20, 5))
    state.scaler = scaler
    state.model_config = {"sequence_length": 10}

    return state


def _make_mock_df(n=100):
    """Build a fake OHLCV dataframe."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.random.rand(n) * 100 + 100,
            "High": np.random.rand(n) * 100 + 110,
            "Low": np.random.rand(n) * 100 + 90,
            "Close": np.random.rand(n) * 100 + 100,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )
    return df


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _make_app(state):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_state] = lambda: state
    return app


class TestPredictEndpoint:
    def test_predict_model_not_ready(self):
        state = _make_mock_state(ready=False)
        client = TestClient(_make_app(state))
        resp = client.post("/predict", json={"horizon": 5})
        assert resp.status_code == 503

    def test_predict_without_uncertainty(self):
        state = _make_mock_state()
        df = _make_mock_df(100)
        client = TestClient(_make_app(state))

        with patch("src.api.routers.predict.load_data_from_db", return_value=df):
            resp = client.post("/predict", json={"horizon": 3, "with_uncertainty": False})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert data["forecast_horizon"] == 3
        assert data["last_known_price"] > 0

    def test_predict_with_uncertainty(self):
        state = _make_mock_state()
        df = _make_mock_df(100)
        client = TestClient(_make_app(state))

        with patch("src.api.routers.predict.load_data_from_db", return_value=df):
            resp = client.post(
                "/predict",
                json={"horizon": 3, "with_uncertainty": True, "n_samples": 10, "confidence_level": 0.95},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        # Should have bounds
        for p in data["predictions"]:
            assert p["lower_bound"] is not None
            assert p["upper_bound"] is not None

    def test_predict_internal_error(self):
        state = _make_mock_state()
        client = TestClient(_make_app(state))
        with patch("src.api.routers.predict.load_data_from_db", side_effect=RuntimeError("db error")):
            resp = client.post("/predict", json={"horizon": 3})
        assert resp.status_code == 500


class TestInferenceEndpoint:
    def test_inference_model_not_ready(self):
        state = _make_mock_state(ready=False)
        client = TestClient(_make_app(state))
        resp = client.post("/predict/inference", json={"sequence": [100.0] * 10, "steps": 3})
        assert resp.status_code == 503

    def test_inference_1d_input(self):
        state = _make_mock_state()
        client = TestClient(_make_app(state))
        resp = client.post("/predict/inference", json={"sequence": [100.0] * 10, "steps": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert data["input_length"] == 10

    def test_inference_2d_input_rejected(self):
        """2D input should be rejected by schema (List[float] expected)."""
        state = _make_mock_state()
        seq_2d = [[100, 110, 90, 105, 5000000]] * 10
        client = TestClient(_make_app(state))
        resp = client.post("/predict/inference", json={"sequence": seq_2d, "steps": 2})
        assert resp.status_code == 422

    def test_inference_internal_error(self):
        state = _make_mock_state()
        state.scaler.transform = MagicMock(side_effect=ValueError("bad input"))
        client = TestClient(_make_app(state))
        resp = client.post("/predict/inference", json={"sequence": [100.0] * 10, "steps": 1})
        assert resp.status_code == 500
