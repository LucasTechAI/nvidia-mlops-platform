"""Tests for the ReAct agent tools module."""


class TestToolRegistry:
    """Tests for tool registration and availability."""

    def test_tool_registry_exists(self):
        """TOOL_REGISTRY should be importable and non-empty."""
        from src.agent.tools import TOOL_REGISTRY

        assert isinstance(TOOL_REGISTRY, dict)
        assert len(TOOL_REGISTRY) >= 4

    def test_tool_registry_has_required_tools(self):
        """All 4 required tools should be registered."""
        from src.agent.tools import TOOL_REGISTRY

        required = ["query_stock_data", "predict_stock_prices", "get_model_metrics", "search_documents"]
        for tool_name in required:
            assert tool_name in TOOL_REGISTRY, f"Missing tool: {tool_name}"

    def test_tool_registry_structure(self):
        """Each tool entry should have name, description, function."""
        from src.agent.tools import TOOL_REGISTRY

        for name, info in TOOL_REGISTRY.items():
            assert "name" in info, f"Tool {name} missing 'name'"
            assert "description" in info, f"Tool {name} missing 'description'"
            assert "function" in info, f"Tool {name} missing 'function'"
            assert callable(info["function"]), f"Tool {name} function not callable"


class TestSearchDocuments:
    """Tests for the search_documents tool."""

    def test_search_documents_returns_string(self):
        """search_documents should return a string."""
        from src.agent.tools import TOOL_REGISTRY

        result = TOOL_REGISTRY["search_documents"]["function"]("NVIDIA LSTM model")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_documents_relevant_results(self):
        """search_documents should return relevant content."""
        from src.agent.tools import TOOL_REGISTRY

        result = TOOL_REGISTRY["search_documents"]["function"]("LSTM architecture")
        # Should contain something about the model
        assert any(term in result.lower() for term in ["lstm", "model", "architecture", "neural"])


class TestReActAgent:
    """Tests for the ReActAgent class."""

    def test_agent_creation(self):
        """Agent should be creatable without errors."""
        from src.agent.react_agent import ReActAgent

        agent = ReActAgent(temperature=0.1)
        assert agent is not None
        assert agent.temperature == 0.1
        assert len(agent.tools) >= 4

    def test_create_agent_factory(self):
        """create_agent factory should work."""
        from src.agent.react_agent import create_agent

        agent = create_agent(temperature=0.5)
        assert agent is not None
        assert agent.temperature == 0.5

    def test_agent_has_tool_descriptions(self):
        """Agent should have formatted tool descriptions."""
        from src.agent.react_agent import ReActAgent

        agent = ReActAgent()
        assert len(agent.tool_descriptions) > 0
        assert "query_stock_data" in agent.tool_descriptions


class TestRAGPipeline:
    """Tests for the RAG pipeline."""

    def test_rag_pipeline_creation(self):
        """RAG pipeline should be creatable."""
        from src.agent.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        assert pipeline is not None

    def test_rag_knowledge_documents(self):
        """RAG should have knowledge documents."""
        from src.agent.rag_pipeline import KNOWLEDGE_DOCUMENTS

        assert len(KNOWLEDGE_DOCUMENTS) >= 5

    def test_rag_retrieve_returns_list(self):
        """retrieve() should return a list of strings."""
        from src.agent.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.index_documents()
        results = pipeline.retrieve("NVIDIA LSTM model")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("content" in r for r in results)

    def test_rag_singleton(self):
        """get_rag_pipeline should return a singleton."""
        from src.agent.rag_pipeline import get_rag_pipeline

        p1 = get_rag_pipeline()
        p2 = get_rag_pipeline()
        assert p1 is p2
