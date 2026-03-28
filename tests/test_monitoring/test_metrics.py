"""Tests for the Prometheus metrics module."""

import pytest

from src.monitoring.metrics import (
    PROMETHEUS_AVAILABLE,
    get_metrics,
    track_agent_query,
    track_prediction,
    track_request,
)


class TestGetMetrics:
    def test_returns_bytes(self):
        result = get_metrics()
        assert isinstance(result, bytes)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_contains_metric_names(self):
        # Track some events first
        track_request("GET", "/health", 200, 0.01)
        track_prediction(True, 0.5)
        data = get_metrics().decode()
        assert "http_requests_total" in data
        assert "predictions_total" in data


class TestTrackRequest:
    def test_track_request_no_error(self):
        """Should not raise even if prometheus is a stub."""
        track_request("POST", "/predict", 200, 0.123)

    def test_track_error_request(self):
        track_request("GET", "/data", 500, 1.0)


class TestTrackPrediction:
    def test_track_success(self):
        track_prediction(True, 0.5)

    def test_track_failure(self):
        track_prediction(False, 0.1)


class TestTrackAgentQuery:
    def test_track_with_tools(self):
        track_agent_query(True, 2.5, ["search_documents", "get_stock_info"])

    def test_track_empty_tools(self):
        track_agent_query(False, 0.5, [])


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
class TestMetricObjects:
    def test_request_count_labels(self):
        from src.monitoring.metrics import REQUEST_COUNT
        REQUEST_COUNT.labels(method="GET", endpoint="/test", status_code="200").inc()

    def test_prediction_latency_observe(self):
        from src.monitoring.metrics import PREDICTION_LATENCY
        PREDICTION_LATENCY.observe(0.42)

    def test_model_loaded_gauge(self):
        from src.monitoring.metrics import MODEL_LOADED
        MODEL_LOADED.set(1)

    def test_drift_score_gauge(self):
        from src.monitoring.metrics import DRIFT_SCORE
        DRIFT_SCORE.set(0.15)

    def test_llm_token_usage(self):
        from src.monitoring.metrics import LLM_TOKEN_USAGE
        LLM_TOKEN_USAGE.labels(type="prompt").inc(100)
        LLM_TOKEN_USAGE.labels(type="completion").inc(50)
