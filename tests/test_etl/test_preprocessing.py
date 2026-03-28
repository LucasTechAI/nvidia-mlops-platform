"""Tests for ETL preprocessing module."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch

from src.etl.preprocessing import (
    StockDataset,
    normalize_features,
    create_sequences,
    train_val_test_split,
    create_data_loaders,
    get_last_sequence,
    inverse_transform,
)


# ---------------------------------------------------------------------------
# Tests — StockDataset
# ---------------------------------------------------------------------------

class TestStockDataset:
    def test_len(self):
        X = np.random.randn(50, 10, 3).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        ds = StockDataset(X, y)
        assert len(ds) == 50

    def test_getitem_returns_tensors(self):
        X = np.random.randn(10, 5, 2).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        ds = StockDataset(X, y)
        seq, target = ds[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)

    def test_1d_target_expanded(self):
        X = np.random.randn(10, 5, 1).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)  # 1D
        ds = StockDataset(X, y)
        _, target = ds[0]
        assert target.dim() == 1  # should be (1,) after unsqueeze


# ---------------------------------------------------------------------------
# Tests — normalize_features (ETL version)
# ---------------------------------------------------------------------------

class TestNormalizeFeatures:
    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "Open": np.random.randn(100) * 10 + 100,
            "Close": np.random.randn(100) * 10 + 100,
        })

    def test_minmax_range(self, df):
        data, scaler = normalize_features(df, ["Open", "Close"])
        assert data.min() >= -1e-6
        assert data.max() <= 1.0 + 1e-6

    def test_standard_scaler(self, df):
        data, scaler = normalize_features(df, ["Open"], scaler_type="StandardScaler")
        # StandardScaler should center around 0
        assert abs(data.mean()) < 0.5

    def test_invalid_scaler_raises(self, df):
        with pytest.raises(ValueError, match="Unknown scaler"):
            normalize_features(df, ["Open"], scaler_type="FooScaler")

    def test_pre_fitted_scaler(self, df):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df[["Open"]].values)
        data, _ = normalize_features(df, ["Open"], fit_scaler=False, scaler=scaler)
        assert data.shape == (100, 1)

    def test_no_scaler_no_fit_raises(self, df):
        with pytest.raises(ValueError, match="Must provide scaler"):
            normalize_features(df, ["Open"], fit_scaler=False, scaler=None)


# ---------------------------------------------------------------------------
# Tests — create_sequences (ETL version)
# ---------------------------------------------------------------------------

class TestCreateSequences:
    def test_shapes(self):
        data = np.random.randn(200, 3).astype(np.float32)
        X, y = create_sequences(data, sequence_length=30)
        assert X.shape == (170, 30, 3)
        assert y.shape == (170, 1)

    def test_too_short_data_raises(self):
        data = np.random.randn(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="must be greater"):
            create_sequences(data, sequence_length=10)


# ---------------------------------------------------------------------------
# Tests — train_val_test_split (ETL version)
# ---------------------------------------------------------------------------

class TestTrainValTestSplit:
    def test_default_split(self):
        X = np.random.randn(100, 10, 3)
        y = np.random.randn(100, 1)
        splits = train_val_test_split(X, y)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = sum(len(v[0]) for v in splits.values())
        assert total == 100

    def test_bad_ratios_raise(self):
        X = np.random.randn(50, 5, 2)
        y = np.random.randn(50, 1)
        with pytest.raises(AssertionError):
            train_val_test_split(X, y, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


# ---------------------------------------------------------------------------
# Tests — create_data_loaders
# ---------------------------------------------------------------------------

class TestCreateDataLoaders:
    def test_loaders_created(self):
        splits = {
            "train": (np.random.randn(50, 10, 3).astype(np.float32), np.random.randn(50, 1).astype(np.float32)),
            "val": (np.random.randn(10, 10, 3).astype(np.float32), np.random.randn(10, 1).astype(np.float32)),
            "test": (np.random.randn(10, 10, 3).astype(np.float32), np.random.randn(10, 1).astype(np.float32)),
        }
        loaders = create_data_loaders(splits, batch_size=16)
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_batches_yield_tensors(self):
        splits = {
            "train": (np.random.randn(20, 5, 2).astype(np.float32), np.random.randn(20, 1).astype(np.float32)),
        }
        loaders = create_data_loaders(splits, batch_size=8)
        for batch_x, batch_y in loaders["train"]:
            assert isinstance(batch_x, torch.Tensor)
            assert isinstance(batch_y, torch.Tensor)
            break


# ---------------------------------------------------------------------------
# Tests — get_last_sequence
# ---------------------------------------------------------------------------

class TestGetLastSequence:
    def test_output_shape(self):
        from sklearn.preprocessing import MinMaxScaler

        df = pd.DataFrame({"Close": np.random.randn(100) * 10 + 100})
        scaler = MinMaxScaler()
        scaler.fit(df[["Close"]].values)
        seq = get_last_sequence(df, scaler, sequence_length=10, feature_columns=["Close"])
        assert seq.shape == (1, 10, 1)


# ---------------------------------------------------------------------------
# Tests — inverse_transform (ETL version)
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_1d_input(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = np.random.randn(50, 1).astype(np.float32)
        scaler.fit(data)
        norm = scaler.transform(data).flatten()
        result = inverse_transform(norm, scaler, n_features=1)
        np.testing.assert_allclose(result, data.flatten(), atol=1e-5)

    def test_2d_input(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = np.random.randn(50, 3).astype(np.float32)
        scaler.fit(data)
        norm = scaler.transform(data)
        result = inverse_transform(norm, scaler)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_3d_input(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_2d = np.random.randn(100, 3).astype(np.float32)
        scaler.fit(data_2d)
        data_3d = data_2d.reshape(10, 10, 3)
        norm_3d = scaler.transform(data_3d.reshape(-1, 3)).reshape(10, 10, 3)
        result = inverse_transform(norm_3d, scaler)
        assert result.shape == (10, 10, 3)
