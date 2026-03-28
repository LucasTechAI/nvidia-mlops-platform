"""LLM telemetry integration (Langfuse / TruLens).

Tracks LLM interactions for observability, debugging, and evaluation:
    - Faithfulness
    - Relevancy
    - Generation latency
    - Token usage
    - Trace/span hierarchy

Referência: Langfuse — Open-source LLM observability
            https://langfuse.com/docs
"""

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TelemetryTracer:
    """LLM telemetry tracer with Langfuse integration.

    Provides tracing for LLM calls, tool invocations, and
    RAG pipeline stages. Falls back to local logging if
    Langfuse is not configured.

    Attributes:
        langfuse: Langfuse client (or None if not available).
        traces: In-memory trace storage (fallback).
    """

    def __init__(self):
        """Initialize the telemetry tracer."""
        self._langfuse = None
        self.traces: list[dict] = []
        self._enabled = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"

        if self._enabled:
            self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Initialize Langfuse client if credentials are available."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse telemetry initialized (host=%s)", host)
            except ImportError:
                logger.info("Langfuse not installed. Using local trace storage.")
            except Exception as e:
                logger.warning("Langfuse init failed: %s. Using local traces.", str(e))
        else:
            logger.info("Langfuse credentials not set. Using local trace storage.")

    def trace_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        latency: float,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record an LLM call.

        Args:
            model: Model name (e.g., "gpt-4o-mini").
            prompt: Input prompt text.
            response: LLM response text.
            latency: Call duration in seconds.
            tokens_prompt: Number of input tokens.
            tokens_completion: Number of output tokens.
            metadata: Additional metadata.

        Returns:
            Trace ID.
        """
        trace_id = f"llm-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.traces)}"

        trace_data = {
            "trace_id": trace_id,
            "type": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency_seconds": round(latency, 4),
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_prompt + tokens_completion,
            "metadata": metadata or {},
        }

        self.traces.append(trace_data)

        if self._langfuse:
            try:
                trace = self._langfuse.trace(name="llm_call", metadata=metadata or {})
                trace.generation(
                    name="generation",
                    model=model,
                    input=prompt[:500],  # Truncate for storage
                    output=response[:500],
                    usage={
                        "input": tokens_prompt,
                        "output": tokens_completion,
                        "total": tokens_prompt + tokens_completion,
                    },
                    metadata={"latency": latency},
                )
            except Exception as e:
                logger.debug("Langfuse trace failed: %s", str(e))

        # Update Prometheus metrics
        try:
            from src.monitoring.metrics import LLM_TOKEN_USAGE

            LLM_TOKEN_USAGE.labels(type="prompt").inc(tokens_prompt)
            LLM_TOKEN_USAGE.labels(type="completion").inc(tokens_completion)
        except Exception:
            pass

        logger.debug(
            "LLM trace: model=%s, latency=%.2fs, tokens=%d",
            model,
            latency,
            tokens_prompt + tokens_completion,
        )

        return trace_id

    def trace_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        latency: float,
        success: bool = True,
        parent_trace_id: Optional[str] = None,
    ) -> str:
        """Record a tool invocation.

        Args:
            tool_name: Name of the tool.
            tool_input: Input provided to the tool.
            tool_output: Output returned by the tool.
            latency: Call duration in seconds.
            success: Whether the tool call succeeded.
            parent_trace_id: ID of the parent LLM trace.

        Returns:
            Trace ID.
        """
        trace_id = f"tool-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.traces)}"

        trace_data = {
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "type": "tool_call",
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "input_length": len(tool_input),
            "output_length": len(tool_output),
            "latency_seconds": round(latency, 4),
            "success": success,
        }

        self.traces.append(trace_data)

        if self._langfuse and parent_trace_id:
            try:
                self._langfuse.span(
                    name=f"tool:{tool_name}",
                    input=tool_input[:300],
                    output=tool_output[:300],
                    metadata={"success": success, "latency": latency},
                )
            except Exception:
                pass

        return trace_id

    def trace_rag_retrieval(
        self,
        query: str,
        n_results: int,
        latency: float,
        contexts: Optional[list[str]] = None,
    ) -> str:
        """Record a RAG retrieval operation.

        Args:
            query: Search query.
            n_results: Number of results retrieved.
            latency: Retrieval duration in seconds.
            contexts: Retrieved context snippets.

        Returns:
            Trace ID.
        """
        trace_id = f"rag-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.traces)}"

        trace_data = {
            "trace_id": trace_id,
            "type": "rag_retrieval",
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],
            "n_results": n_results,
            "latency_seconds": round(latency, 4),
        }

        self.traces.append(trace_data)
        return trace_id

    @contextmanager
    def trace_span(self, name: str, metadata: Optional[dict] = None):
        """Context manager for tracing a code span.

        Usage:
            with tracer.trace_span("my_operation") as span:
                # ... do work ...
                span["result"] = "success"
        """
        span: dict[str, Any] = {
            "name": name,
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        start = time.time()
        try:
            yield span
        finally:
            span["latency_seconds"] = round(time.time() - start, 4)
            span["end_time"] = datetime.now().isoformat()
            self.traces.append({"type": "span", **span})

    def get_summary(self) -> dict:
        """Get summary of all recorded traces.

        Returns:
            Summary with counts, latencies, token usage.
        """
        llm_traces = [t for t in self.traces if t.get("type") == "llm_call"]
        tool_traces = [t for t in self.traces if t.get("type") == "tool_call"]
        rag_traces = [t for t in self.traces if t.get("type") == "rag_retrieval"]

        total_tokens = sum(t.get("tokens_total", 0) for t in llm_traces)
        total_latency = sum(t.get("latency_seconds", 0) for t in llm_traces)

        return {
            "total_traces": len(self.traces),
            "llm_calls": len(llm_traces),
            "tool_calls": len(tool_traces),
            "rag_retrievals": len(rag_traces),
            "total_tokens": total_tokens,
            "total_llm_latency": round(total_latency, 2),
            "avg_llm_latency": round(total_latency / len(llm_traces), 3) if llm_traces else 0,
            "langfuse_connected": self._langfuse is not None,
        }

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception as e:
                logger.warning("Langfuse flush failed: %s", str(e))


# Singleton instance
_tracer: Optional[TelemetryTracer] = None


def get_tracer() -> TelemetryTracer:
    """Get or create the global telemetry tracer."""
    global _tracer
    if _tracer is None:
        _tracer = TelemetryTracer()
    return _tracer
