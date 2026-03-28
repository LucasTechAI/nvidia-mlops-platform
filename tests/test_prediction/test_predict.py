"""Tests for the prediction module."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import NvidiaLSTM
from src.prediction.predict import (
    calculate_prediction_intervals,
    generate_forecast,
    inverse_transform_predictions,
    save_predictions_to_csv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model_5feat():
    """Trained-ish model with 5 input features."""
    model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, output_size=5)
    model.eval()
    return model


@pytest.fixture
def last_sequence():
    """A single sequence of shape (seq_len, n_features)."""
    np.random.seed(42)
    return np.random.randn(10, 5).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests — generate_forecast
# ---------------------------------------------------------------------------


class TestGenerateForecast:
    def test_output_shape(self, model_5feat, last_sequence, device):
        horizon = 7
        preds = generate_forecast(model_5feat, last_sequence, horizon=horizon, device=device)
        assert preds.shape == (horizon, 5)

    def test_single_step(self, model_5feat, last_sequence, device):
        preds = generate_forecast(model_5feat, last_sequence, horizon=1, device=device)
        assert preds.shape == (1, 5)

    def test_large_horizon(self, model_5feat, last_sequence, device):
        preds = generate_forecast(model_5feat, last_sequence, horizon=60, device=device)
        assert preds.shape == (60, 5)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Tests — inverse_transform_predictions
# ---------------------------------------------------------------------------


class TestInverseTransformPredictions:
    def test_inverse_transform(self, tmp_path):
        import pickle

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        data = np.random.randn(100, 5)
        scaler.fit(data)
        scaler_path = str(tmp_path / "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        normalized = scaler.transform(data[:10])
        result = inverse_transform_predictions(normalized, scaler_path)
        np.testing.assert_allclose(result, data[:10], atol=1e-6)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            inverse_transform_predictions(np.zeros((5, 3)), "/nonexistent/scaler.pkl")


# ---------------------------------------------------------------------------
# Tests — calculate_prediction_intervals
# ---------------------------------------------------------------------------


class TestCalculatePredictionIntervals:
    def test_interval_shapes(self):
        preds = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        lower, upper = calculate_prediction_intervals(preds, confidence_level=0.95)
        assert lower.shape == preds.shape
        assert upper.shape == preds.shape

    def test_bounds_ordering(self):
        preds = np.array([100.0, 200.0, 300.0])
        lower, upper = calculate_prediction_intervals(preds, confidence_level=0.95)
        assert np.all(lower <= preds)
        assert np.all(upper >= preds)

    def test_wider_interval_for_higher_confidence(self):
        preds = np.array([100.0, 200.0, 300.0])
        lower90, upper90 = calculate_prediction_intervals(preds, confidence_level=0.90)
        lower99, upper99 = calculate_prediction_intervals(preds, confidence_level=0.99)
        width90 = upper90 - lower90
        width99 = upper99 - lower99
        assert np.all(width99 > width90)

    def test_custom_uncertainty_factor(self):
        preds = np.array([100.0])
        lower_10, upper_10 = calculate_prediction_intervals(preds, uncertainty_factor=0.10)
        lower_20, upper_20 = calculate_prediction_intervals(preds, uncertainty_factor=0.20)
        width_10 = (upper_10 - lower_10)[0]
        width_20 = (upper_20 - lower_20)[0]
        assert width_20 > width_10


# ---------------------------------------------------------------------------
# Tests — save_predictions_to_csv
# ---------------------------------------------------------------------------


class TestSavePredictionsCsv:
    def test_save_1d(self, tmp_path):
        import pandas as pd

        preds = np.array([1.0, 2.0, 3.0])
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        path = str(tmp_path / "preds.csv")
        save_predictions_to_csv(preds, dates, path)
        df = pd.read_csv(path)
        assert len(df) == 3
        assert "Date" in df.columns
        assert "Prediction" in df.columns

    def test_save_2d(self, tmp_path):
        import pandas as pd

        preds = np.array([[1.0, 2.0], [3.0, 4.0]])
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        path = str(tmp_path / "preds2d.csv")
        save_predictions_to_csv(preds, dates, path, column_names=["Open", "Close"])
        df = pd.read_csv(path)
        assert len(df) == 2
        assert "Open" in df.columns
        assert "Close" in df.columns
