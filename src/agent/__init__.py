"""ReAct agent with custom tools and RAG pipeline.

Provides a financial domain AI assistant for NVIDIA stock analysis
using a ReAct loop with 4 tools: stock query, LSTM prediction,
metrics analysis, and RAG document search.
"""

from src.agent.react_agent import ReActAgent, create_agent

__all__ = ["ReActAgent", "create_agent"]
