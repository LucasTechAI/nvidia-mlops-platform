"""Tests for agent/rag_pipeline.py — RAGPipeline and utility functions."""

from unittest.mock import MagicMock, patch

from src.agent.rag_pipeline import (
    KNOWLEDGE_DOCUMENTS,
    RAGPipeline,
    retrieve_context,
)


class TestRAGPipelineInit:
    def test_defaults(self):
        p = RAGPipeline()
        assert p.collection_name == "nvidia_knowledge"
        assert "chroma_db" in p.persist_directory

    def test_custom_params(self):
        p = RAGPipeline(collection_name="test_col", persist_directory="/tmp/test_chroma")
        assert p.collection_name == "test_col"
        assert p.persist_directory == "/tmp/test_chroma"


class TestMemorySearch:
    """Test the fallback keyword search (no ChromaDB dependency)."""

    def test_memory_search_returns_results(self):
        p = RAGPipeline()
        results = p._memory_search("NVIDIA stock prediction LSTM", top_k=3)
        assert len(results) <= 3
        assert all("content" in r for r in results)

    def test_memory_search_relevance(self):
        p = RAGPipeline()
        results = p._memory_search("LSTM model architecture", top_k=1)
        assert len(results) == 1
        # Should find the LSTM methodology doc
        assert "LSTM" in results[0]["content"]

    def test_memory_search_top_k(self):
        p = RAGPipeline()
        results = p._memory_search("nvidia", top_k=2)
        assert len(results) == 2


class TestIndexDocuments:
    """Test indexing with ChromaDB mocked out."""

    def test_index_without_chromadb_uses_memory(self):
        p = RAGPipeline()
        p._client = None
        p._collection = None
        # Force ChromaDB to be unavailable
        with patch.object(p, "_get_client", return_value=None):
            count = p.index_documents()
        assert count == len(KNOWLEDGE_DOCUMENTS)
        assert hasattr(p, "_memory_docs")

    def test_index_custom_documents(self):
        p = RAGPipeline()
        with patch.object(p, "_get_client", return_value=None):
            docs = [{"id": "test1", "content": "hello world", "metadata": {}}]
            count = p.index_documents(docs)
        assert count == 1

    def test_index_with_mock_collection(self):
        p = RAGPipeline()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        p._collection = mock_collection

        count = p.index_documents()
        assert count == len(KNOWLEDGE_DOCUMENTS)
        mock_collection.add.assert_called_once()

    def test_index_skips_existing(self):
        p = RAGPipeline()
        mock_collection = MagicMock()
        existing_ids = [d["id"] for d in KNOWLEDGE_DOCUMENTS]
        mock_collection.get.return_value = {"ids": existing_ids}
        p._collection = mock_collection

        count = p.index_documents()
        assert count == 0


class TestRetrieve:
    def test_retrieve_with_mock_collection(self):
        p = RAGPipeline()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["doc content A", "doc content B"]],
            "metadatas": [[{"source": "kb"}, {"source": "kb"}]],
            "distances": [[0.1, 0.5]],
        }
        p._collection = mock_collection

        results = p.retrieve("NVIDIA stock", top_k=2)
        assert len(results) == 2
        assert results[0]["content"] == "doc content A"
        assert results[0]["distance"] == 0.1

    def test_retrieve_falls_back_to_memory(self):
        p = RAGPipeline()
        p._collection = None
        with patch.object(p, "_get_collection", return_value=None):
            results = p.retrieve("NVIDIA LSTM model", top_k=2)
        assert len(results) <= 2

    def test_retrieve_handles_exception(self):
        p = RAGPipeline()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.side_effect = RuntimeError("db error")
        p._collection = mock_collection

        # Should fall back to memory search
        results = p.retrieve("test query", top_k=2)
        assert isinstance(results, list)


class TestIndexGoldenSet:
    def test_golden_set_missing_file(self):
        p = RAGPipeline()
        with patch("src.agent.rag_pipeline.PROJECT_ROOT", MagicMock()):
            # Make the path not exist
            from pathlib import Path
            with patch.object(Path, "exists", return_value=False):
                count = p.index_golden_set()
        # Should return 0 when file doesn't exist
        assert count == 0


class TestRetrieveContext:
    def test_retrieve_context_returns_string(self):
        with patch("src.agent.rag_pipeline.get_rag_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.retrieve.return_value = [
                {"content": "doc1 content", "metadata": {"source": "kb", "topic": "model"}, "distance": 0.1},
            ]
            mock_get.return_value = mock_pipeline

            ctx = retrieve_context("NVIDIA stock", top_k=1)
            assert "doc1 content" in ctx
            assert "[Source 1:" in ctx

    def test_retrieve_context_empty(self):
        with patch("src.agent.rag_pipeline.get_rag_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.retrieve.return_value = []
            mock_get.return_value = mock_pipeline

            ctx = retrieve_context("something", top_k=1)
            assert ctx == ""


class TestKnowledgeDocuments:
    def test_knowledge_documents_not_empty(self):
        assert len(KNOWLEDGE_DOCUMENTS) >= 5

    def test_all_docs_have_required_fields(self):
        for doc in KNOWLEDGE_DOCUMENTS:
            assert "id" in doc
            assert "content" in doc
            assert "metadata" in doc
            assert len(doc["content"]) > 20
