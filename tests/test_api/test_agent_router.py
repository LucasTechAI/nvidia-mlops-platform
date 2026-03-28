"""Tests for api/routers/agent.py endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.agent import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestAgentQueryEndpoint:
    def test_query_success(self, client):
        mock_agent = MagicMock()
        mock_agent.model_name = "test-model"
        mock_agent.query_with_guardrails.return_value = {
            "answer": "NVIDIA stock is at $130.",
            "tools_used": ["query_stock_data"],
            "iterations": 2,
        }
        with patch("src.agent.react_agent.create_agent", return_value=mock_agent):
            resp = client.post("/agent/query", json={"query": "NVIDIA stock price?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["guardrails_applied"] is True

    def test_query_without_guardrails(self, client):
        mock_agent = MagicMock()
        mock_agent.model_name = "test-model"
        mock_agent.query.return_value = {
            "answer": "Answer",
            "tools_used": [],
            "iterations": 1,
        }
        with patch("src.agent.react_agent.create_agent", return_value=mock_agent):
            resp = client.post(
                "/agent/query",
                json={"query": "NVIDIA stock price?", "use_guardrails": False},
            )
        assert resp.status_code == 200
        assert resp.json()["guardrails_applied"] is False

    def test_query_agent_error(self, client):
        with patch("src.agent.react_agent.create_agent", side_effect=RuntimeError("LLM unavailable")):
            resp = client.post("/agent/query", json={"query": "NVIDIA stock?"})
        assert resp.status_code == 500


class TestAgentHealthEndpoint:
    def test_agent_health_success(self, client):
        mock_agent = MagicMock()
        mock_agent.llm_provider = "openai"
        mock_agent.model_name = "gpt-4"
        with (
            patch("src.agent.react_agent.create_agent", return_value=mock_agent),
            patch.dict("src.agent.tools.TOOL_REGISTRY", {"query_stock_data": MagicMock()}),
        ):
            resp = client.get("/agent/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_agent_health_degraded(self, client):
        with patch("src.agent.react_agent.create_agent", side_effect=RuntimeError("no LLM")):
            resp = client.get("/agent/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "degraded" in data["status"]
