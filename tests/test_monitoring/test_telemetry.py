"""Tests for the telemetry module."""

import time

import pytest

from src.monitoring.telemetry import TelemetryTracer, get_tracer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracer(monkeypatch):
    """Tracer with telemetry disabled (no Langfuse)."""
    monkeypatch.setenv("TELEMETRY_ENABLED", "false")
    return TelemetryTracer()


@pytest.fixture
def tracer_enabled(monkeypatch):
    """Tracer enabled but without Langfuse creds (local fallback)."""
    monkeypatch.setenv("TELEMETRY_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    return TelemetryTracer()


# ---------------------------------------------------------------------------
# Tests — initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_disabled_tracer(self, tracer):
        assert tracer._langfuse is None
        assert tracer._enabled is False
        assert tracer.traces == []

    def test_enabled_without_creds(self, tracer_enabled):
        assert tracer_enabled._langfuse is None
        assert tracer_enabled._enabled is True


# ---------------------------------------------------------------------------
# Tests — trace_llm_call
# ---------------------------------------------------------------------------


class TestTraceLlmCall:
    def test_returns_trace_id(self, tracer_enabled):
        tid = tracer_enabled.trace_llm_call(
            model="gpt-4o-mini",
            prompt="Hello",
            response="Hi there",
            latency=0.5,
            tokens_prompt=5,
            tokens_completion=3,
        )
        assert tid.startswith("llm-")

    def test_stores_trace(self, tracer_enabled):
        tracer_enabled.trace_llm_call(model="test", prompt="p", response="r", latency=0.1)
        assert len(tracer_enabled.traces) == 1
        assert tracer_enabled.traces[0]["type"] == "llm_call"
        assert tracer_enabled.traces[0]["model"] == "test"

    def test_token_totals(self, tracer_enabled):
        tracer_enabled.trace_llm_call(
            model="m",
            prompt="p",
            response="r",
            latency=0.1,
            tokens_prompt=100,
            tokens_completion=50,
        )
        assert tracer_enabled.traces[0]["tokens_total"] == 150

    def test_metadata_stored(self, tracer_enabled):
        tracer_enabled.trace_llm_call(
            model="m",
            prompt="p",
            response="r",
            latency=0.1,
            metadata={"key": "value"},
        )
        assert tracer_enabled.traces[0]["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Tests — trace_tool_call
# ---------------------------------------------------------------------------


class TestTraceToolCall:
    def test_returns_trace_id(self, tracer_enabled):
        tid = tracer_enabled.trace_tool_call(
            tool_name="search",
            tool_input="query",
            tool_output="result",
            latency=0.2,
        )
        assert tid.startswith("tool-")

    def test_stores_trace(self, tracer_enabled):
        tracer_enabled.trace_tool_call(
            tool_name="calc",
            tool_input="2+2",
            tool_output="4",
            latency=0.01,
            success=True,
        )
        assert len(tracer_enabled.traces) == 1
        assert tracer_enabled.traces[0]["tool_name"] == "calc"
        assert tracer_enabled.traces[0]["success"] is True

    def test_failed_tool(self, tracer_enabled):
        tracer_enabled.trace_tool_call(
            tool_name="broken",
            tool_input="x",
            tool_output="error",
            latency=0.5,
            success=False,
        )
        assert tracer_enabled.traces[0]["success"] is False


# ---------------------------------------------------------------------------
# Tests — trace_rag_retrieval
# ---------------------------------------------------------------------------


class TestTraceRagRetrieval:
    def test_returns_trace_id(self, tracer_enabled):
        tid = tracer_enabled.trace_rag_retrieval(
            query="what is NVDA?",
            n_results=5,
            latency=0.3,
        )
        assert tid.startswith("rag-")

    def test_stores_trace(self, tracer_enabled):
        tracer_enabled.trace_rag_retrieval(
            query="q",
            n_results=3,
            latency=0.1,
            contexts=["c1", "c2"],
        )
        assert tracer_enabled.traces[0]["type"] == "rag_retrieval"
        assert tracer_enabled.traces[0]["n_results"] == 3


# ---------------------------------------------------------------------------
# Tests — trace_span
# ---------------------------------------------------------------------------


class TestTraceSpan:
    def test_context_manager(self, tracer_enabled):
        with tracer_enabled.trace_span("test_op") as span:
            span["result"] = "ok"
            time.sleep(0.01)

        assert len(tracer_enabled.traces) == 1
        assert tracer_enabled.traces[0]["name"] == "test_op"
        assert tracer_enabled.traces[0]["latency_seconds"] >= 0.01


# ---------------------------------------------------------------------------
# Tests — get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_empty_summary(self, tracer_enabled):
        summary = tracer_enabled.get_summary()
        assert summary["total_traces"] == 0
        assert summary["llm_calls"] == 0
        assert summary["langfuse_connected"] is False

    def test_summary_with_traces(self, tracer_enabled):
        tracer_enabled.trace_llm_call("m", "p", "r", 0.5, 10, 5)
        tracer_enabled.trace_llm_call("m", "p", "r", 0.3, 20, 10)
        tracer_enabled.trace_tool_call("t", "i", "o", 0.1)
        summary = tracer_enabled.get_summary()
        assert summary["total_traces"] == 3
        assert summary["llm_calls"] == 2
        assert summary["tool_calls"] == 1
        assert summary["total_tokens"] == 45
        assert summary["avg_llm_latency"] > 0


# ---------------------------------------------------------------------------
# Tests — get_tracer singleton
# ---------------------------------------------------------------------------


class TestGetTracer:
    def test_returns_tracer(self, monkeypatch):
        monkeypatch.setenv("TELEMETRY_ENABLED", "false")
        # Reset singleton
        import src.monitoring.telemetry as mod

        mod._tracer = None
        t = get_tracer()
        assert isinstance(t, TelemetryTracer)

    def test_singleton(self, monkeypatch):
        monkeypatch.setenv("TELEMETRY_ENABLED", "false")
        import src.monitoring.telemetry as mod

        mod._tracer = None
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
