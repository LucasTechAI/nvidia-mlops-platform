"""Tests for the monitoring modules."""

import numpy as np
import pandas as pd


class TestDriftDetection:
    """Tests for drift detection."""

    def test_calculate_psi_identical(self):
        """PSI of identical distributions should be ~0."""
        from src.monitoring.drift import calculate_psi

        np.random.seed(42)
        data = np.random.randn(1000)
        psi = calculate_psi(data, data)
        assert psi < 0.01

    def test_calculate_psi_different(self):
        """PSI of different distributions should be > 0."""
        from src.monitoring.drift import calculate_psi

        np.random.seed(42)
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 5  # Shifted by 5
        psi = calculate_psi(ref, cur)
        assert psi > 0.2  # Significant drift

    def test_calculate_psi_non_negative(self):
        """PSI should always be non-negative."""
        from src.monitoring.drift import calculate_psi

        np.random.seed(42)
        ref = np.random.randn(500)
        cur = np.random.randn(500) * 2
        psi = calculate_psi(ref, cur)
        assert psi >= 0

    def test_detect_drift_no_drift(self):
        """Same data should show no drift."""
        from src.monitoring.drift import detect_drift

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Open": np.random.randn(200) + 100,
                "High": np.random.randn(200) + 105,
                "Low": np.random.randn(200) + 95,
                "Close": np.random.randn(200) + 100,
                "Volume": np.random.randint(1000, 10000, 200).astype(float),
            }
        )
        result = detect_drift(df, df, save_results=False)
        assert result["drift_detected"] is False
        assert result["overall_status"] == "no_drift"

    def test_detect_drift_with_shift(self):
        """Shifted data should be detected as drift."""
        from src.monitoring.drift import detect_drift

        np.random.seed(42)
        ref = pd.DataFrame(
            {
                "Close": np.random.randn(500) + 100,
            }
        )
        cur = pd.DataFrame(
            {
                "Close": np.random.randn(500) + 200,  # Big shift
            }
        )
        result = detect_drift(ref, cur, features=["Close"], save_results=False)
        assert result["drift_detected"] is True

    def test_detect_drift_result_structure(self):
        """Result should have expected keys."""
        from src.monitoring.drift import detect_drift

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Close": np.random.randn(100) + 100,
            }
        )
        result = detect_drift(df, df, features=["Close"], save_results=False)
        assert "timestamp" in result
        assert "features" in result
        assert "overall_status" in result
        assert "drift_detected" in result
        assert "retrain_recommended" in result


class TestPrometheusMetrics:
    """Tests for Prometheus metrics module."""

    def test_metrics_import(self):
        """Metrics module should be importable."""
        from src.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None

    def test_get_metrics(self):
        """get_metrics should return bytes."""
        from src.monitoring.metrics import get_metrics

        result = get_metrics()
        assert isinstance(result, bytes)

    def test_track_request(self):
        """track_request should not raise."""
        from src.monitoring.metrics import track_request

        track_request("GET", "/health", 200, 0.05)

    def test_track_prediction(self):
        """track_prediction should not raise."""
        from src.monitoring.metrics import track_prediction

        track_prediction(True, 0.5)

    def test_track_agent_query(self):
        """track_agent_query should not raise."""
        from src.monitoring.metrics import track_agent_query

        track_agent_query(True, 2.0, ["query_stock_data"])


class TestTelemetry:
    """Tests for LLM telemetry module."""

    def test_tracer_creation(self):
        """Tracer should be creatable."""
        from src.monitoring.telemetry import TelemetryTracer

        tracer = TelemetryTracer()
        assert tracer is not None

    def test_trace_llm_call(self):
        """trace_llm_call should return trace ID."""
        from src.monitoring.telemetry import TelemetryTracer

        tracer = TelemetryTracer()
        trace_id = tracer.trace_llm_call(
            model="gpt-4o-mini",
            prompt="test prompt",
            response="test response",
            latency=1.5,
            tokens_prompt=10,
            tokens_completion=20,
        )
        assert trace_id.startswith("llm-")
        assert len(tracer.traces) == 1

    def test_trace_tool_call(self):
        """trace_tool_call should return trace ID."""
        from src.monitoring.telemetry import TelemetryTracer

        tracer = TelemetryTracer()
        trace_id = tracer.trace_tool_call(
            tool_name="query_stock_data",
            tool_input="test input",
            tool_output="test output",
            latency=0.5,
        )
        assert trace_id.startswith("tool-")

    def test_get_summary(self):
        """get_summary should return valid summary."""
        from src.monitoring.telemetry import TelemetryTracer

        tracer = TelemetryTracer()
        tracer.trace_llm_call("model", "p", "r", 1.0, 10, 20)
        tracer.trace_tool_call("tool", "i", "o", 0.5)
        summary = tracer.get_summary()
        assert summary["total_traces"] == 2
        assert summary["llm_calls"] == 1
        assert summary["tool_calls"] == 1
        assert summary["total_tokens"] == 30

    def test_get_tracer_singleton(self):
        """get_tracer should return same instance."""
        from src.monitoring.telemetry import get_tracer

        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
