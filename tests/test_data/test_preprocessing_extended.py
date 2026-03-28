"""Additional tests for data preprocessing (normalization, sequences, splits)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    normalize_features,
    create_sequences,
    train_val_test_split,
    load_scaler,
    inverse_transform,
)


# ---------------------------------------------------------------------------
# Tests — normalize_features
# ---------------------------------------------------------------------------

class TestNormalizeFeatures:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "Open": np.random.randn(100) * 10 + 100,
            "Close": np.random.randn(100) * 10 + 100,
            "Volume": np.random.randint(1e6, 1e7, 100).astype(float),
        })

    def test_output_range(self, sample_df):
        data, scaler = normalize_features(sample_df, ["Open", "Close"])
        assert data.min() >= -1e-6
        assert data.max() <= 1.0 + 1e-6

    def test_output_shape(self, sample_df):
        data, scaler = normalize_features(sample_df, ["Open", "Close", "Volume"])
        assert data.shape == (100, 3)

    def test_scaler_saved(self, sample_df, tmp_path):
        path = str(tmp_path / "scaler.pkl")
        _, _ = normalize_features(sample_df, ["Open"], scaler_path=path)
        assert Path(path).exists()

    def test_unsupported_scaler_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported"):
            normalize_features(sample_df, ["Open"], scaler_type="StandardScaler")


# ---------------------------------------------------------------------------
# Tests — create_sequences
# ---------------------------------------------------------------------------

class TestCreateSequences:
    def test_shapes(self):
        data = np.random.randn(200, 3)
        X, y = create_sequences(data, sequence_length=30)
        assert X.shape == (170, 30, 3)
        assert y.shape == (170, 3)

    def test_single_feature(self):
        data = np.random.randn(100, 1)
        X, y = create_sequences(data, sequence_length=10)
        assert X.shape[1] == 10
        assert X.shape[2] == 1

    def test_forecast_horizon(self):
        data = np.random.randn(100, 2)
        X, y = create_sequences(data, sequence_length=10, forecast_horizon=5)
        expected_n = 100 - 10 - 5 + 1
        assert X.shape[0] == expected_n

    def test_content_correctness(self):
        """Verify the first sequence and target are correct."""
        data = np.arange(20).reshape(20, 1).astype(float)
        X, y = create_sequences(data, sequence_length=5)
        np.testing.assert_array_equal(X[0], data[:5])
        np.testing.assert_array_equal(y[0], data[5])


# ---------------------------------------------------------------------------
# Tests — train_val_test_split
# ---------------------------------------------------------------------------

class TestTrainValTestSplit:
    @pytest.fixture
    def data(self):
        X = np.random.randn(100, 10, 3)
        y = np.random.randn(100, 3)
        return X, y

    def test_default_split_sizes(self, data):
        X, y = data
        X_tr, y_tr, X_v, y_v, X_te, y_te = train_val_test_split(X, y)
        assert len(X_tr) == 70
        assert len(X_v) == 15
        assert len(X_te) == 15
        assert len(X_tr) + len(X_v) + len(X_te) == 100

    def test_custom_split(self, data):
        X, y = data
        X_tr, y_tr, X_v, y_v, X_te, y_te = train_val_test_split(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        assert len(X_tr) == 80
        assert len(X_v) == 10

    def test_invalid_ratios_raise(self, data):
        X, y = data
        with pytest.raises(ValueError, match="sum to 1.0"):
            train_val_test_split(X, y, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_temporal_order_preserved(self, data):
        X, y = data
        # Mark each sample with its index
        X_indexed = np.arange(100).reshape(100, 1, 1).repeat(10, axis=1).repeat(3, axis=2).astype(float)
        X_tr, _, X_v, _, X_te, _ = train_val_test_split(X_indexed, y)
        # All train indices < all val indices < all test indices
        assert X_tr[-1, 0, 0] < X_v[0, 0, 0]
        assert X_v[-1, 0, 0] < X_te[0, 0, 0]


# ---------------------------------------------------------------------------
# Tests — load_scaler / inverse_transform
# ---------------------------------------------------------------------------

class TestScalerUtils:
    def test_load_scaler(self, tmp_path):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit(np.random.randn(50, 3))
        path = str(tmp_path / "scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

        loaded = load_scaler(path)
        assert hasattr(loaded, "inverse_transform")

    def test_inverse_transform_roundtrip(self):
        from sklearn.preprocessing import MinMaxScaler

        data = np.random.randn(20, 3)
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        restored = inverse_transform(normalized, scaler)
        np.testing.assert_allclose(restored, data, atol=1e-6)
