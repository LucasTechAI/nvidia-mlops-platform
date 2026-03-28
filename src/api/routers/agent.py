"""Agent router for the ReAct financial assistant.

Provides endpoints to interact with the ReAct agent for
NVIDIA stock analysis, predictions, and document search.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


# ============== Schemas ==============


class AgentQueryRequest(BaseModel):
    """Request to query the ReAct agent."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or instruction for the agent",
        examples=["Qual foi o preço de fechamento da NVIDIA hoje?"],
    )
    use_guardrails: bool = Field(
        default=True,
        description="Apply input/output guardrails (prompt injection, PII)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM sampling temperature",
    )
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=15,
        description="Maximum reasoning-action iterations",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Override LLM model name",
    )


class AgentQueryResponse(BaseModel):
    """Response from the ReAct agent."""

    answer: str
    reasoning_steps: int
    tools_used: list[str]
    elapsed_time: float
    guardrails_applied: bool
    model_used: str


class AgentHealthResponse(BaseModel):
    """Agent health check response."""

    status: str
    llm_provider: str
    model_name: str
    tools_available: list[str]
    rag_indexed: bool


# ============== Endpoints ==============


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(request: AgentQueryRequest):
    """Send a question to the ReAct agent.

    The agent uses reasoning-action loops with tools to answer
    financial questions about NVIDIA stock data.
    """
    try:
        from src.agent.react_agent import create_agent

        agent = create_agent(
            model_name=request.model_name,
            temperature=request.temperature,
            max_iterations=request.max_iterations,
        )

        start_time = time.time()

        if request.use_guardrails:
            result = agent.query_with_guardrails(request.query)
        else:
            result = agent.query(request.query)

        elapsed = time.time() - start_time

        # Parse tools used from result
        tools_used = result.get("tools_used", [])
        reasoning_steps = result.get("iterations", 0)

        return AgentQueryResponse(
            answer=result.get("answer", "No answer generated."),
            reasoning_steps=reasoning_steps,
            tools_used=tools_used,
            elapsed_time=round(elapsed, 3),
            guardrails_applied=request.use_guardrails,
            model_used=agent.model_name,
        )

    except Exception as e:
        logger.error("Agent query failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent query failed: {str(e)}",
        ) from e


@router.get("/health", response_model=AgentHealthResponse)
async def agent_health():
    """Check the health and configuration of the agent."""
    try:
        from src.agent.react_agent import create_agent
        from src.agent.tools import TOOL_REGISTRY

        agent = create_agent()

        # Check RAG status
        rag_indexed = False
        try:
            from src.agent.rag_pipeline import get_rag_pipeline

            pipeline = get_rag_pipeline()
            rag_indexed = pipeline.is_indexed
        except Exception:
            pass

        return AgentHealthResponse(
            status="healthy",
            llm_provider=agent.llm_provider,
            model_name=agent.model_name,
            tools_available=list(TOOL_REGISTRY.keys()),
            rag_indexed=rag_indexed,
        )
    except Exception as e:
        logger.error("Agent health check failed: %s", str(e))
        return AgentHealthResponse(
            status=f"degraded: {str(e)}",
            llm_provider="unknown",
            model_name="unknown",
            tools_available=[],
            rag_indexed=False,
        )
