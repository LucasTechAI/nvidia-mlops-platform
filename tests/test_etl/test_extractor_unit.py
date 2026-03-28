"""Unit tests for etl/extractor_nvidia.py (no network calls)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestExtractNvidiaDataUnit:
    def test_extract_returns_dataframe(self):
        from src.etl.extractor_nvidia import extract_nvidia_data

        mock_ticker = MagicMock()
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5),
                "Open": [100, 101, 102, 103, 104],
                "High": [110, 111, 112, 113, 114],
                "Low": [90, 91, 92, 93, 94],
                "Close": [105, 106, 107, 108, 109],
                "Volume": [1e6, 2e6, 3e6, 4e6, 5e6],
            }
        )
        mock_ticker.history.return_value = mock_data
        with patch("src.etl.extractor_nvidia.Ticker", return_value=mock_ticker):
            result = extract_nvidia_data(period="5d", interval="1d")
        assert isinstance(result, pd.DataFrame)
        assert "Date" in result.columns

    def test_extract_empty_returns_none(self):
        from src.etl.extractor_nvidia import extract_nvidia_data

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("src.etl.extractor_nvidia.Ticker", return_value=mock_ticker):
            result = extract_nvidia_data()
        assert result is None

    def test_extract_renames_datetime_column(self):
        from src.etl.extractor_nvidia import extract_nvidia_data

        mock_ticker = MagicMock()
        mock_data = pd.DataFrame(
            {
                "Datetime": pd.date_range("2024-01-01", periods=3),
                "Close": [100, 101, 102],
            }
        )
        mock_ticker.history.return_value = mock_data
        with patch("src.etl.extractor_nvidia.Ticker", return_value=mock_ticker):
            result = extract_nvidia_data()
        assert "Date" in result.columns
        assert "Datetime" not in result.columns


class TestSaveData:
    def test_save_to_absolute_path(self):
        from src.etl.extractor_nvidia import save_data

        df = pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_data.csv")
            save_data(df, path=path)
            assert Path(path).exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == 1

    def test_save_creates_directories(self):
        from src.etl.extractor_nvidia import save_data

        df = pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "test.csv")
            save_data(df, path=path)
            assert Path(path).exists()


class TestShowStatistics:
    def test_runs_without_error(self):
        from src.etl.extractor_nvidia import show_statistics

        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": range(10),
                "High": range(10, 20),
                "Low": range(10),
                "Close": range(5, 15),
                "Volume": range(1000, 1010),
            }
        )
        # Should not raise
        show_statistics(df)
