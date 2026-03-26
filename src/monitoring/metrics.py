"""Prometheus custom metrics for FastAPI instrumentation.

Metrics:
    - Request latency (histogram)
    - Request count by endpoint (counter)
    - Active requests (gauge)
    - Prediction errors (counter)
    - Model inference latency (histogram)
    - Agent query latency (histogram)
    - Token usage (counter)
"""

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Try to import prometheus_client, provide fallback stubs
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
    registry = CollectorRegistry()

    # HTTP request metrics
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
        registry=registry,
    )

    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=registry,
    )

    ACTIVE_REQUESTS = Gauge(
        "http_active_requests",
        "Number of active HTTP requests",
        ["method", "endpoint"],
        registry=registry,
    )

    # Prediction metrics
    PREDICTION_COUNT = Counter(
        "predictions_total",
        "Total prediction requests",
        ["status"],
        registry=registry,
    )

    PREDICTION_LATENCY = Histogram(
        "prediction_duration_seconds",
        "Prediction latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=registry,
    )

    PREDICTION_ERRORS = Counter(
        "prediction_errors_total",
        "Total prediction errors",
        ["error_type"],
        registry=registry,
    )

    # Agent metrics
    AGENT_QUERY_COUNT = Counter(
        "agent_queries_total",
        "Total agent queries",
        ["status"],
        registry=registry,
    )

    AGENT_QUERY_LATENCY = Histogram(
        "agent_query_duration_seconds",
        "Agent query latency in seconds",
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        registry=registry,
    )

    AGENT_TOOLS_USED = Counter(
        "agent_tools_used_total",
        "Total tool invocations by the agent",
        ["tool_name"],
        registry=registry,
    )

    # Model metrics
    MODEL_LOADED = Gauge(
        "model_loaded",
        "Whether the model is currently loaded (1=yes, 0=no)",
        registry=registry,
    )

    DRIFT_SCORE = Gauge(
        "data_drift_psi_score",
        "Current PSI drift score",
        registry=registry,
    )

    # Token usage
    LLM_TOKEN_USAGE = Counter(
        "llm_token_usage_total",
        "Total LLM tokens used",
        ["type"],  # prompt, completion
        registry=registry,
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be no-ops.")

    # Stub classes
    class _StubMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

    registry = None
    REQUEST_COUNT = _StubMetric()
    REQUEST_LATENCY = _StubMetric()
    ACTIVE_REQUESTS = _StubMetric()
    PREDICTION_COUNT = _StubMetric()
    PREDICTION_LATENCY = _StubMetric()
    PREDICTION_ERRORS = _StubMetric()
    AGENT_QUERY_COUNT = _StubMetric()
    AGENT_QUERY_LATENCY = _StubMetric()
    AGENT_TOOLS_USED = _StubMetric()
    MODEL_LOADED = _StubMetric()
    DRIFT_SCORE = _StubMetric()
    LLM_TOKEN_USAGE = _StubMetric()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output.

    Returns:
        Prometheus text format metrics as bytes.
    """
    if PROMETHEUS_AVAILABLE and registry is not None:
        return generate_latest(registry)
    return b"# Prometheus not available\n"


def track_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Track an HTTP request."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def track_prediction(success: bool, duration: float) -> None:
    """Track a prediction request."""
    status = "success" if success else "error"
    PREDICTION_COUNT.labels(status=status).inc()
    PREDICTION_LATENCY.observe(duration)


def track_agent_query(success: bool, duration: float, tools: list[str]) -> None:
    """Track an agent query."""
    status = "success" if success else "error"
    AGENT_QUERY_COUNT.labels(status=status).inc()
    AGENT_QUERY_LATENCY.observe(duration)
    for tool in tools:
        AGENT_TOOLS_USED.labels(tool_name=tool).inc()


def create_metrics_middleware() -> Callable:
    """Create a FastAPI middleware for automatic request tracking.

    Returns:
        ASGI middleware function.
    """

    async def metrics_middleware(request, call_next):
        method = request.method
        endpoint = request.url.path

        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            track_request(method, endpoint, response.status_code, duration)
            return response
        except Exception as e:
            duration = time.time() - start_time
            track_request(method, endpoint, 500, duration)
            raise e
        finally:
            ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()

    return metrics_middleware
