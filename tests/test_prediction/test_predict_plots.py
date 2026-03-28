"""Extended tests for predict module (plot and utility functions)."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.predict import (
    plot_predictions,
    plot_predictions_with_intervals,
)


@pytest.fixture
def historical_df():
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": np.random.randn(200).cumsum() + 500,
        }
    )


@pytest.fixture
def forecast_data():
    np.random.seed(42)
    return np.random.randn(30).cumsum() + 500


@pytest.fixture
def forecast_dates():
    return pd.date_range("2023-07-20", periods=30, freq="D")


class TestPlotPredictions:
    def test_saves_plot(self, historical_df, forecast_data, forecast_dates, tmp_path):
        path = str(tmp_path / "pred.png")
        plot_predictions(historical_df, forecast_data, forecast_dates, save_path=path)
        from pathlib import Path

        assert Path(path).exists()

    def test_no_save(self, historical_df, forecast_data, forecast_dates):
        # Should not raise
        plot_predictions(historical_df, forecast_data, forecast_dates)


class TestPlotPredictionsWithIntervals:
    def test_saves_plot(self, historical_df, forecast_data, forecast_dates, tmp_path):
        lower = forecast_data - 10
        upper = forecast_data + 10
        path = str(tmp_path / "pred_ci.png")
        plot_predictions_with_intervals(
            historical_df,
            forecast_data,
            forecast_dates,
            lower,
            upper,
            save_path=path,
        )
        from pathlib import Path

        assert Path(path).exists()
