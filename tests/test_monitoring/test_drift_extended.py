"""Tests for monitoring/drift.py — detect_drift_from_db and _run_evidently_report."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    _run_evidently_report,
    detect_drift,
    detect_drift_from_db,
)


class TestDetectDriftFromDb:
    """Tests for the convenience function detect_drift_from_db."""

    def _create_db(self, n=100):
        tmpdir = tempfile.mkdtemp()
        db_path = str(Path(tmpdir) / "test.db")
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": np.random.rand(n) * 10 + 100,
                "High": np.random.rand(n) * 10 + 110,
                "Low": np.random.rand(n) * 10 + 90,
                "Close": np.random.rand(n) * 10 + 100,
                "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
            }
        )
        conn = sqlite3.connect(db_path)
        df.to_sql("nvidia_stock", conn, index=False)
        conn.close()
        return db_path

    def test_from_db_success(self):
        db_path = self._create_db(100)
        with patch("src.config.DATABASE_PATH", db_path):
            result = detect_drift_from_db(train_ratio=0.7)
        assert "overall_status" in result
        assert "features" in result

    def test_from_db_missing_database(self):
        with patch("src.config.DATABASE_PATH", "/nonexistent/db.sqlite"):
            result = detect_drift_from_db()
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_from_db_insufficient_data(self):
        db_path = self._create_db(10)
        # make it only 10 rows — split gives < 10 per side
        with patch("src.config.DATABASE_PATH", db_path):
            result = detect_drift_from_db(train_ratio=0.7)
        # With 10 rows, 7 ref + 3 current; feature needs >=10 each
        # Some features may be skipped
        assert isinstance(result, dict)

    def test_from_db_query_failure(self):
        tmpdir = tempfile.mkdtemp()
        db_path = str(Path(tmpdir) / "empty.db")
        # Create empty DB (no nvidia_stock table)
        conn = sqlite3.connect(db_path)
        conn.close()
        with patch("src.config.DATABASE_PATH", db_path):
            result = detect_drift_from_db()
        assert result["status"] == "error"
        assert "failed" in result["message"].lower()


class TestDetectDriftEvidentlyBranch:
    """Test the Evidently branch in detect_drift (lines 154-155, 177-205)."""

    def test_evidently_import_error(self):
        """When evidently is not importable, PSI-only mode."""
        ref = pd.DataFrame({"Close": np.random.rand(100) * 10 + 100})
        cur = pd.DataFrame({"Close": np.random.rand(100) * 10 + 100})
        with patch("src.monitoring.drift._run_evidently_report", side_effect=ImportError("no evidently")):
            result = detect_drift(ref, cur, features=["Close"], save_results=False)
        assert "features" in result
        assert "evidently_report" not in result

    def test_evidently_runtime_error(self):
        """When evidently fails at runtime, we still get PSI results."""
        ref = pd.DataFrame({"Close": np.random.rand(100) * 10 + 100})
        cur = pd.DataFrame({"Close": np.random.rand(100) * 10 + 100})
        with patch("src.monitoring.drift._run_evidently_report", side_effect=RuntimeError("boom")):
            result = detect_drift(ref, cur, features=["Close"], save_results=False)
        assert "features" in result

    def test_no_common_features(self):
        ref = pd.DataFrame({"A": [1, 2, 3]})
        cur = pd.DataFrame({"B": [4, 5, 6]})
        result = detect_drift(ref, cur, features=["X", "Y"], save_results=False)
        assert result["status"] == "error"

    def test_insufficient_feature_data(self):
        """Features with fewer than 10 samples get skipped."""
        ref = pd.DataFrame({"Close": [1.0] * 5})
        cur = pd.DataFrame({"Close": [2.0] * 5})
        result = detect_drift(ref, cur, features=["Close"], save_results=False)
        # Feature skipped, no PSI scores
        assert len(result["features"]) == 0


class TestRunEvidentlyReport:
    """Test _run_evidently_report directly — evidently not installed."""

    def test_report_raises_import_error_without_evidently(self):
        """Without evidently installed, calling _run_evidently_report raises ImportError."""
        ref = pd.DataFrame({"Close": np.random.rand(50)})
        cur = pd.DataFrame({"Close": np.random.rand(50)})
        with pytest.raises(ImportError):
            _run_evidently_report(ref, cur, ["Close"])
