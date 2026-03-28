"""Extended tests for monitoring/metrics.py — tracking functions and middleware."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.metrics import (
    create_metrics_middleware,
    get_metrics,
    track_agent_query,
    track_prediction,
    track_request,
)


class TestGetMetrics:
    def test_returns_bytes(self):
        result = get_metrics()
        assert isinstance(result, bytes)

    def test_non_empty(self):
        result = get_metrics()
        assert len(result) > 0


class TestTrackRequest:
    def test_success(self):
        # Should not raise
        track_request("GET", "/health", 200, 0.05)

    def test_various_methods(self):
        track_request("POST", "/predict", 200, 0.1)
        track_request("GET", "/status", 404, 0.02)
        track_request("PUT", "/update", 500, 1.0)


class TestTrackPrediction:
    def test_success(self):
        track_prediction(success=True, duration=0.5)

    def test_error(self):
        track_prediction(success=False, duration=0.1)


class TestTrackAgentQuery:
    def test_success_with_tools(self):
        track_agent_query(success=True, duration=2.0, tools=["search", "predict"])

    def test_success_no_tools(self):
        track_agent_query(success=True, duration=1.0, tools=[])

    def test_error(self):
        track_agent_query(success=False, duration=0.5, tools=["search"])


class TestCreateMetricsMiddleware:
    def test_returns_callable(self):
        middleware = create_metrics_middleware()
        assert callable(middleware)

    def test_middleware_success(self):
        middleware = create_metrics_middleware()
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/health"

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_call_next(request):
            return mock_response

        async def run():
            return await middleware(mock_request, mock_call_next)

        result = asyncio.run(run())
        assert result.status_code == 200

    def test_middleware_exception(self):
        middleware = create_metrics_middleware()
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/predict"

        async def mock_call_next(request):
            raise ValueError("test error")

        async def run():
            await middleware(mock_request, mock_call_next)

        with pytest.raises(ValueError, match="test error"):
            asyncio.run(run())
