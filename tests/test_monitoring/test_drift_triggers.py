"""Tests for the multi-trigger retrain system: staleness + CI breach + combined."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    CI_BREACH_RATIO_THRESHOLD,
    check_model_staleness,
    check_prediction_breach,
    detect_all_triggers,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trigger 2: Model Staleness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckModelStaleness:
    """Tests for check_model_staleness."""

    def test_fresh_model_not_stale(self, tmp_path):
        """A model created just now should NOT be stale."""
        model_file = tmp_path / "fresh_model.pth"
        model_file.write_bytes(b"fake model")
        result = check_model_staleness(str(model_file), max_age_days=30)
        assert result["stale"] is False
        assert result["age_days"] < 1
        assert result["threshold_days"] == 30

    def test_old_model_is_stale(self, tmp_path):
        """A model older than threshold should be stale."""
        import os
        import time

        model_file = tmp_path / "old_model.pth"
        model_file.write_bytes(b"fake model")
        # Set mtime to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(str(model_file), (old_time, old_time))

        result = check_model_staleness(str(model_file), max_age_days=30)
        assert result["stale"] is True
        assert result["age_days"] >= 59  # ~60 days
        assert "reason" in result

    def test_missing_model_is_stale(self):
        """A missing model file should be treated as stale."""
        result = check_model_staleness("/nonexistent/model.pth")
        assert result["stale"] is True
        assert result["age_days"] == float("inf")
        assert "not found" in result.get("reason", "").lower()

    def test_exactly_at_threshold(self, tmp_path):
        """A model exactly at the threshold boundary (30 days)."""
        import os
        import time

        model_file = tmp_path / "boundary_model.pth"
        model_file.write_bytes(b"fake model")
        boundary_time = time.time() - (30 * 86400)
        os.utime(str(model_file), (boundary_time, boundary_time))

        result = check_model_staleness(str(model_file), max_age_days=30)
        assert result["stale"] is True

    def test_custom_threshold(self, tmp_path):
        """Test with a custom threshold of 7 days."""
        import os
        import time

        model_file = tmp_path / "week_old_model.pth"
        model_file.write_bytes(b"fake model")
        old_time = time.time() - (10 * 86400)  # 10 days
        os.utime(str(model_file), (old_time, old_time))

        result_7 = check_model_staleness(str(model_file), max_age_days=7)
        assert result_7["stale"] is True

        result_14 = check_model_staleness(str(model_file), max_age_days=14)
        assert result_14["stale"] is False  # 10 days < 14 threshold

        result_30 = check_model_staleness(str(model_file), max_age_days=30)
        assert result_30["stale"] is False  # 10 days < 30 threshold

    def test_default_model_path(self):
        """Test that default path is used when model_path is None."""
        result = check_model_staleness(model_path=None)
        # Should work without error (model may or may not exist)
        assert "stale" in result

    def test_result_has_last_modified(self, tmp_path):
        """Result should include ISO-formatted last_modified timestamp."""
        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"fake model")
        result = check_model_staleness(str(model_file))
        assert result["last_modified"] is not None
        # Should be ISO format
        datetime.fromisoformat(result["last_modified"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trigger 3: Prediction CI Breach
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckPredictionBreach:
    """Tests for check_prediction_breach."""

    def test_no_breach_good_predictions(self):
        """Accurate predictions → no breach."""
        np.random.seed(42)
        predictions = np.array([100, 102, 98, 101, 99, 103, 97, 100, 102, 98] * 10)
        # Actuals very close to predictions
        actuals = predictions + np.random.randn(100) * 0.5

        result = check_prediction_breach(predictions, actuals)
        assert result["breach_detected"] is False
        assert result["breach_ratio"] < CI_BREACH_RATIO_THRESHOLD

    def test_breach_with_large_errors(self):
        """Wildly wrong predictions → breach."""
        np.random.seed(42)
        predictions = np.ones(100) * 100
        # Actuals way off
        actuals = np.ones(100) * 200

        result = check_prediction_breach(predictions, actuals)
        assert result["breach_detected"] is True
        assert result["breach_ratio"] > CI_BREACH_RATIO_THRESHOLD
        assert "concept drift" in result.get("reason", "").lower()

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            check_prediction_breach(np.array([1, 2, 3]), np.array([1, 2]))

    def test_too_few_observations(self):
        """Less than 5 observations → not enough data, no breach."""
        result = check_prediction_breach(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert result["breach_detected"] is False
        assert "insufficient" in result.get("reason", "").lower()

    def test_perfect_predictions_zero_breach(self):
        """Perfect predictions → 0% breach (within floating point)."""
        predictions = np.arange(50, 60, 0.2)
        actuals = predictions.copy()  # Exact match
        result = check_prediction_breach(predictions, actuals)
        assert result["breach_detected"] is False
        assert result["breach_ratio"] == 0.0
        assert result["n_breaches"] == 0

    def test_custom_confidence_level(self):
        """Narrower CI (80%) should catch more breaches."""
        np.random.seed(42)
        predictions = np.ones(100) * 100
        actuals = predictions + np.random.randn(100) * 5

        result_95 = check_prediction_breach(predictions, actuals, confidence_level=0.95)
        result_80 = check_prediction_breach(predictions, actuals, confidence_level=0.80)

        # Narrower CI → more breaches
        assert result_80["breach_ratio"] >= result_95["breach_ratio"]

    def test_custom_breach_threshold(self):
        """Custom threshold should change the decision."""
        np.random.seed(42)
        predictions = np.ones(100) * 100
        actuals = predictions + np.random.randn(100) * 8

        result_strict = check_prediction_breach(predictions, actuals, breach_threshold=0.01)
        result_relaxed = check_prediction_breach(predictions, actuals, breach_threshold=0.99)

        assert result_strict["breach_detected"] is True
        assert result_relaxed["breach_detected"] is False

    def test_result_fields_complete(self):
        """All expected fields should be present."""
        np.random.seed(42)
        predictions = np.random.randn(50) * 10 + 100
        actuals = predictions + np.random.randn(50)

        result = check_prediction_breach(predictions, actuals)
        assert "breach_detected" in result
        assert "breach_ratio" in result
        assert "breach_threshold" in result
        assert "confidence_level" in result
        assert "residual_std" in result
        assert "mean_residual" in result
        assert "ci_margin" in result
        assert "n_breaches" in result
        assert "n_total" in result

    def test_biased_predictions_detected(self):
        """Systematically biased predictions → high breach ratio."""
        predictions = np.ones(100) * 100
        actuals = np.ones(100) * 120  # Consistent 20-point bias
        result = check_prediction_breach(predictions, actuals)
        # With 0 variance in residuals (all same bias), std is 0 and all are outside CI
        assert result["breach_detected"] is True
        assert result["mean_residual"] == 20.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Combined: detect_all_triggers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDetectAllTriggers:
    """Tests for the combined multi-trigger detection."""

    @pytest.fixture
    def fresh_model(self, tmp_path):
        """Create a fresh model file."""
        model_file = tmp_path / "fresh_model.pth"
        model_file.write_bytes(b"fake model")
        return str(model_file)

    @pytest.fixture
    def stale_model(self, tmp_path):
        """Create a stale model file (60 days old)."""
        import os
        import time

        model_file = tmp_path / "stale_model.pth"
        model_file.write_bytes(b"fake model")
        old_time = time.time() - (60 * 86400)
        os.utime(str(model_file), (old_time, old_time))
        return str(model_file)

    @pytest.fixture
    def stable_data(self):
        """Reference and current data from the same distribution."""
        np.random.seed(42)
        n = 200
        ref = pd.DataFrame({"Close": np.random.randn(n) * 10 + 100})
        np.random.seed(99)
        cur = pd.DataFrame({"Close": np.random.randn(n) * 10 + 100})
        return ref, cur

    def test_all_healthy(self, tmp_path, fresh_model, stable_data, monkeypatch):
        """No triggers → healthy."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data
        preds = np.random.randn(50) * 10 + 100
        actuals = preds + np.random.randn(50) * 0.5

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            predictions=preds,
            actuals=actuals,
            model_path=fresh_model,
            save_results=True,
        )
        assert result["retrain_recommended"] is False
        assert result["overall_status"] == "healthy"
        assert len(result["active_triggers"]) == 0

    def test_staleness_only_trigger(self, tmp_path, stale_model, stable_data, monkeypatch):
        """Only staleness fires → retrain recommended."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            model_path=stale_model,
            save_results=False,
        )
        assert result["retrain_recommended"] is True
        assert "model_staleness" in result["active_triggers"]

    def test_breach_only_trigger(self, tmp_path, fresh_model, stable_data, monkeypatch):
        """Only CI breach fires → retrain recommended."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data
        preds = np.ones(100) * 100
        actuals = np.ones(100) * 200  # Way off

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            predictions=preds,
            actuals=actuals,
            model_path=fresh_model,
            save_results=False,
        )
        assert result["retrain_recommended"] is True
        assert "prediction_ci_breach" in result["active_triggers"]

    def test_no_predictions_skips_breach(self, tmp_path, fresh_model, stable_data, monkeypatch):
        """Without predictions data, CI breach is skipped."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            model_path=fresh_model,
            save_results=False,
        )
        assert result["triggers"]["prediction_breach"]["status"] == "skipped"

    def test_multiple_triggers_fire(self, tmp_path, stale_model, monkeypatch):
        """Multiple triggers firing simultaneously."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        np.random.seed(42)
        n = 200
        ref = pd.DataFrame({"Close": np.random.randn(n) * 10 + 100})
        cur = pd.DataFrame({"Close": np.random.randn(n) * 10 + 500})  # Drifted
        preds = np.ones(100) * 100
        actuals = np.ones(100) * 300  # Way off

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            predictions=preds,
            actuals=actuals,
            model_path=stale_model,
            save_results=False,
        )
        assert result["retrain_recommended"] is True
        assert len(result["active_triggers"]) >= 2

    def test_saves_multi_trigger_report(self, tmp_path, fresh_model, stable_data, monkeypatch):
        """Report is saved to JSON when save_results=True."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data

        detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            model_path=fresh_model,
            save_results=True,
        )
        report_path = tmp_path / "multi_trigger_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "triggers" in data
        assert "retrain_recommended" in data

    def test_summary_message(self, tmp_path, stale_model, stable_data, monkeypatch):
        """Summary message should mention active trigger names."""
        monkeypatch.setattr("src.monitoring.drift.RESULTS_DIR", tmp_path)
        ref, cur = stable_data

        result = detect_all_triggers(
            reference_data=ref,
            current_data=cur,
            model_path=stale_model,
            save_results=False,
        )
        assert "Staleness" in result["summary"]
        assert "trigger" in result["summary"].lower()
