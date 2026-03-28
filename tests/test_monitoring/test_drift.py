"""Tests for the drift detection module."""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    PSI_RETRAIN_THRESHOLD,
    calculate_psi,
    detect_drift,
)

# ---------------------------------------------------------------------------
# Tests — calculate_psi
# ---------------------------------------------------------------------------

class TestCalculatePSI:
    def test_identical_distributions_near_zero(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = calculate_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_high_psi(self):
        np.random.seed(42)
        reference = np.random.randn(1000)
        current = np.random.randn(1000) + 5  # large shift
        psi = calculate_psi(reference, current)
        assert psi > PSI_RETRAIN_THRESHOLD

    def test_psi_non_negative(self):
        np.random.seed(42)
        ref = np.random.randn(500)
        cur = np.random.randn(500) + 0.5
        psi = calculate_psi(ref, cur)
        assert psi >= 0

    def test_psi_symmetric_roughly(self):
        np.random.seed(42)
        a = np.random.randn(500)
        b = np.random.randn(500) + 1
        psi_ab = calculate_psi(a, b)
        psi_ba = calculate_psi(b, a)
        # PSI is not perfectly symmetric but should be in same ballpark
        assert abs(psi_ab - psi_ba) < 0.5

    def test_custom_bins(self):
        np.random.seed(42)
        data = np.random.randn(200)
        psi_5 = calculate_psi(data, data, n_bins=5)
        psi_50 = calculate_psi(data, data, n_bins=50)
        # Both should be near zero for identical data
        assert psi_5 < 0.05
        assert psi_50 < 0.05


# ---------------------------------------------------------------------------
# Tests — detect_drift
# ---------------------------------------------------------------------------

class TestDetectDrift:
    @pytest.fixture
    def reference_df(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "Open": np.random.randn(n) * 10 + 100,
            "High": np.random.randn(n) * 10 + 110,
            "Low": np.random.randn(n) * 10 + 90,
            "Close": np.random.randn(n) * 10 + 100,
            "Volume": np.random.randn(n) * 1e6 + 5e6,
        })

    @pytest.fixture
    def current_no_drift(self, reference_df):
        """Current data from same distribution."""
        np.random.seed(99)
        n = 200
        return pd.DataFrame({
            "Open": np.random.randn(n) * 10 + 100,
            "High": np.random.randn(n) * 10 + 110,
            "Low": np.random.randn(n) * 10 + 90,
            "Close": np.random.randn(n) * 10 + 100,
            "Volume": np.random.randn(n) * 1e6 + 5e6,
        })

    @pytest.fixture
    def current_with_drift(self):
        """Heavily shifted data."""
        np.random.seed(77)
        n = 200
        return pd.DataFrame({
            "Open": np.random.randn(n) * 10 + 500,
            "High": np.random.randn(n) * 10 + 510,
            "Low": np.random.randn(n) * 10 + 490,
            "Close": np.random.randn(n) * 10 + 500,
            "Volume": np.random.randn(n) * 1e6 + 50e6,
        })

    def test_no_drift_detected(self, tmp_path, monkeypatch):
        """Same distribution should yield low PSI."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        np.random.seed(42)
        n = 500
        ref = pd.DataFrame({"Close": np.random.randn(n) * 10 + 100})
        np.random.seed(99)
        cur = pd.DataFrame({"Close": np.random.randn(n) * 10 + 100})
        result = detect_drift(ref, cur, features=["Close"], save_results=True)
        assert result["overall_status"] == "no_drift"

    def test_drift_detected(self, reference_df, current_with_drift, tmp_path, monkeypatch):
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        result = detect_drift(reference_df, current_with_drift, save_results=True)
        assert result["drift_detected"] is True
        assert result["retrain_recommended"] is True

    def test_saves_json_report(self, reference_df, current_no_drift, tmp_path, monkeypatch):
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        detect_drift(reference_df, current_no_drift, save_results=True)
        report_path = tmp_path / "drift_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "features" in data

    def test_no_common_features(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref = pd.DataFrame({"A": [1, 2, 3]})
        cur = pd.DataFrame({"B": [4, 5, 6]})
        result = detect_drift(ref, cur, features=["X"])
        assert result["status"] == "error"

    def test_custom_features(self, reference_df, current_no_drift, tmp_path, monkeypatch):
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        result = detect_drift(
            reference_df, current_no_drift, features=["Close", "Volume"], save_results=False
        )
        assert "Close" in result["features"]
        assert "Volume" in result["features"]
        assert "Open" not in result["features"]

    def test_per_feature_results(self, reference_df, current_with_drift, tmp_path, monkeypatch):
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        result = detect_drift(reference_df, current_with_drift, save_results=False)
        for feat_info in result["features"].values():
            assert "psi" in feat_info
            assert "status" in feat_info
            assert "ref_mean" in feat_info
            assert "cur_mean" in feat_info
