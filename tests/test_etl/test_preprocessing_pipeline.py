"""Tests for etl/preprocessing.py — prepare_data_pipeline, get_last_sequence, inverse_transform."""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.etl.preprocessing import (
    get_last_sequence,
    inverse_transform,
    prepare_data_pipeline,
)


def _create_test_db(n=200, start_year=2020):
    """Create a temporary SQLite database with stock data."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")

    dates = pd.bdate_range(f"{start_year}-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": np.random.rand(n) * 100 + 100,
            "High": np.random.rand(n) * 100 + 110,
            "Low": np.random.rand(n) * 100 + 90,
            "Close": np.random.rand(n) * 100 + 100,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        }
    )

    conn = sqlite3.connect(db_path)
    df.to_sql("nvidia_stock", conn, index=False)
    conn.close()

    return db_path


class TestPrepareDataPipeline:
    """Tests for the full prepare_data_pipeline function."""

    def test_pipeline_returns_loaders_scaler_df(self):
        db_path = _create_test_db(200)
        from src.config import DataConfig

        config = DataConfig(
            feature_columns=["Close"],
            target_column="Close",
            start_year=2020,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            scaler_type="MinMaxScaler",
        )

        loaders, scaler, df = prepare_data_pipeline(
            db_path=db_path,
            data_config=config,
            sequence_length=10,
            batch_size=16,
        )

        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        assert scaler is not None
        assert len(df) > 0

    def test_pipeline_with_multi_feature(self):
        db_path = _create_test_db(200)
        from src.config import DataConfig

        config = DataConfig(
            feature_columns=["Open", "High", "Low", "Close", "Volume"],
            target_column="Close",
            start_year=2020,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            scaler_type="MinMaxScaler",
        )

        loaders, scaler, df = prepare_data_pipeline(
            db_path=db_path,
            data_config=config,
            sequence_length=10,
            batch_size=16,
        )

        # Grab one batch from training loader
        batch = next(iter(loaders["train"]))
        X_batch, y_batch = batch
        assert X_batch.shape[2] == 5  # 5 features


class TestGetLastSequence:
    """Tests for get_last_sequence."""

    def test_returns_correct_shape(self):
        n = 100
        df = pd.DataFrame({"Close": np.random.rand(n) * 100 + 100})
        scaler = MinMaxScaler()
        scaler.fit(df[["Close"]].values)

        result = get_last_sequence(df, scaler, sequence_length=10, feature_columns=["Close"])
        assert result.shape == (1, 10, 1)

    def test_multi_feature(self):
        n = 100
        df = pd.DataFrame(
            {
                "Open": np.random.rand(n) * 100,
                "Close": np.random.rand(n) * 100,
            }
        )
        scaler = MinMaxScaler()
        scaler.fit(df[["Open", "Close"]].values)

        result = get_last_sequence(df, scaler, sequence_length=20, feature_columns=["Open", "Close"])
        assert result.shape == (1, 20, 2)


class TestInverseTransform:
    """Tests for the inverse_transform utility."""

    @pytest.fixture()
    def scaler(self):
        s = MinMaxScaler()
        s.fit(np.array([[0, 0], [100, 100]]))
        return s

    def test_1d_input(self, scaler):
        data = np.array([0.5, 0.5])
        result = inverse_transform(data, scaler, n_features=2)
        assert result.ndim == 1

    def test_2d_input(self, scaler):
        data = np.array([[0.5, 0.5], [0.0, 1.0]])
        result = inverse_transform(data, scaler)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[1], [0, 100], atol=1e-6)

    def test_3d_input(self, scaler):
        data = np.array([[[0.5, 0.5], [1.0, 0.0]]])  # (1, 2, 2)
        result = inverse_transform(data, scaler)
        assert result.shape == (1, 2, 2)
