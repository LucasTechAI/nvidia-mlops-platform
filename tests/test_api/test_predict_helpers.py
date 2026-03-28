"""Extended tests for predict endpoint helper functions."""


import numpy as np
import pandas as pd
import torch

from src.api.routers.predict import (
    generate_forecast_with_uncertainty,
    get_forecast_dates,
)
from src.models.lstm_model import NvidiaLSTM


class TestGetForecastDates:
    def test_skips_weekends(self):
        # Friday Jan 5 2024
        last_date = pd.Timestamp("2024-01-05")
        dates = get_forecast_dates(last_date, horizon=5)
        assert len(dates) == 5
        # First date should be Monday Jan 8
        assert dates[0].weekday() == 0  # Monday

    def test_correct_count(self):
        last_date = pd.Timestamp("2024-01-01")
        dates = get_forecast_dates(last_date, horizon=10)
        assert len(dates) == 10
        # All should be weekdays
        for d in dates:
            assert d.weekday() < 5

    def test_single_day(self):
        last_date = pd.Timestamp("2024-01-01")  # Monday
        dates = get_forecast_dates(last_date, horizon=1)
        assert len(dates) == 1


class TestGenerateForecastWithUncertainty:
    def test_output_shapes(self):
        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.1, output_size=5)
        seq = torch.randn(1, 10, 5)
        horizon = 5
        n_samples = 3
        mean_preds, std_preds = generate_forecast_with_uncertainty(
            model, seq, horizon, n_samples, device="cpu"
        )
        assert mean_preds.shape == (horizon,)
        assert std_preds.shape == (horizon,)

    def test_std_non_negative(self):
        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=2, dropout=0.2, output_size=5)
        seq = torch.randn(1, 10, 5)
        _, std_preds = generate_forecast_with_uncertainty(
            model, seq, horizon=3, n_samples=5, device="cpu"
        )
        assert np.all(std_preds >= 0)

    def test_model_back_to_eval(self):
        model = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.1, output_size=5)
        model.eval()
        seq = torch.randn(1, 10, 5)
        generate_forecast_with_uncertainty(model, seq, horizon=2, n_samples=2, device="cpu")
        # Should be back in eval mode
        assert not model.training
