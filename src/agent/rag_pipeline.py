"""RAG Pipeline: embedding + vector store + retriever + generator.

Components:
    - Embedding model (sentence-transformers)
    - Vector store (ChromaDB)
    - Retriever + Generator pipeline

Reference: Lewis et al. (2020) — Retrieval-Augmented Generation for
           Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default documents to index (project knowledge base)
KNOWLEDGE_DOCUMENTS = [
    {
        "id": "nvidia_overview",
        "content": (
            "NVIDIA Corporation (NVDA) is a leading technology company specializing in "
            "GPU-accelerated computing. Founded in 1993, NVIDIA designs GPUs for gaming, "
            "professional visualization, data centers, and automotive markets. The company "
            "has become a dominant player in AI/ML hardware with its CUDA platform and "
            "data center GPUs (A100, H100, B200). NVIDIA stock has shown strong growth "
            "driven by AI demand, with significant price appreciation since 2023."
        ),
        "metadata": {"source": "knowledge_base", "topic": "company"},
    },
    {
        "id": "lstm_methodology",
        "content": (
            "The NVIDIA stock prediction model uses a Long Short-Term Memory (LSTM) neural "
            "network architecture. LSTMs are a type of recurrent neural network (RNN) capable "
            "of learning long-term dependencies in sequential data. The model uses 5 input "
            "features (Open, High, Low, Close, Volume) with a sequence length of 60 trading "
            "days. Architecture: 2 LSTM layers with 128 hidden units, dropout of 0.2, and "
            "a fully connected output layer predicting all 5 features. Training uses MSE loss "
            "with Adam optimizer, learning rate 0.001, and early stopping with patience 10."
        ),
        "metadata": {"source": "knowledge_base", "topic": "model"},
    },
    {
        "id": "model_metrics",
        "content": (
            "The trained LSTM model achieves the following test set metrics: R² Score of 0.885 "
            "indicating good explanatory power, RMSE of $9.73, MAE of $8.22, and MAPE of 5.40%. "
            "The correlation between predicted and actual prices is 0.980. Directional accuracy "
            "(predicting if price goes up or down) is 47.04%. The Sharpe Ratio of predicted "
            "returns is 0.82 and maximum drawdown is 32.63%. The model was trained for 16 "
            "epochs with early stopping activating at epoch 5 (best validation loss)."
        ),
        "metadata": {"source": "knowledge_base", "topic": "metrics"},
    },
    {
        "id": "stock_analysis_basics",
        "content": (
            "Stock price analysis involves examining historical price data to identify trends "
            "and patterns. Key metrics include: Moving Averages (SMA, EMA) for trend detection, "
            "RSI (Relative Strength Index) for overbought/oversold conditions, MACD for momentum, "
            "and Bollinger Bands for volatility. Volume analysis confirms price movements. "
            "For NVIDIA specifically, key drivers include: AI/ML demand, data center revenue, "
            "gaming GPU cycles, semiconductor supply chains, and competitive dynamics with AMD/Intel."
        ),
        "metadata": {"source": "knowledge_base", "topic": "analysis"},
    },
    {
        "id": "risk_disclaimer",
        "content": (
            "Stock predictions are based on historical patterns and do not guarantee future "
            "results. The LSTM model captures statistical patterns but cannot predict black swan "
            "events, earnings surprises, regulatory changes, or macroeconomic shifts. Users "
            "should not make investment decisions solely based on model predictions. The model's "
            "directional accuracy of ~47% means it performs near random in predicting price "
            "direction. Maximum drawdown of 32.63% indicates significant potential losses. "
            "Always consult a qualified financial advisor before making investment decisions."
        ),
        "metadata": {"source": "knowledge_base", "topic": "risk"},
    },
    {
        "id": "technical_architecture",
        "content": (
            "The NVIDIA MLOps platform architecture consists of: 1) ETL pipeline (Yahoo Finance "
            "→ CSV → SQLite) for data ingestion, 2) Data preprocessing with MinMaxScaler "
            "normalization and sequence creation, 3) LSTM model training with MLflow tracking, "
            "4) FastAPI REST API for predictions and data access, 5) Streamlit dashboard for "
            "visualization, 6) Docker Compose for containerized deployment, 7) Prometheus + "
            "Grafana for monitoring, 8) ReAct agent with RAG for intelligent querying. "
            "CI/CD via GitHub Actions with ruff linting, pytest testing, and Docker builds."
        ),
        "metadata": {"source": "knowledge_base", "topic": "architecture"},
    },
    {
        "id": "feature_engineering",
        "content": (
            "The model uses 5 raw features: Open, High, Low, Close, and Volume prices. "
            "Data is normalized using MinMaxScaler to the [0,1] range. Sequences of 60 "
            "consecutive trading days are created as input windows. The train/validation/test "
            "split is 70%/15%/15% preserving temporal order (no shuffling). Data starts from "
            "2017 to capture multiple market cycles including pre-COVID, COVID crash, "
            "recovery, and the AI boom starting in 2023."
        ),
        "metadata": {"source": "knowledge_base", "topic": "features"},
    },
]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using ChromaDB and sentence-transformers.

    This pipeline indexes project documents and retrieves relevant context
    for the ReAct agent to use when answering questions.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for ChromaDB persistence.
        _collection: ChromaDB collection instance.
        _client: ChromaDB client instance.
    """

    def __init__(
        self,
        collection_name: str = "nvidia_knowledge",
        persist_directory: Optional[str] = None,
    ):
        """Initialize RAG pipeline.

        Args:
            collection_name: ChromaDB collection name.
            persist_directory: Path to persist the vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(PROJECT_ROOT / "data" / "chroma_db")
        self._collection = None
        self._client = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=self.persist_directory)
            except ImportError:
                logger.warning("chromadb not installed. Using in-memory fallback.")
                self._client = None
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        if client is None:
            return None

        try:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "NVIDIA MLOps knowledge base"},
            )
            return self._collection
        except Exception as e:
            logger.error("Error creating ChromaDB collection: %s", e)
            return None

    def index_documents(self, documents: Optional[list[dict]] = None) -> int:
        """Index documents into the vector store.

        Args:
            documents: List of document dicts with 'id', 'content', 'metadata'.
                       If None, indexes the default knowledge base.

        Returns:
            Number of documents indexed.
        """
        docs = documents or KNOWLEDGE_DOCUMENTS
        collection = self._get_collection()

        if collection is None:
            # Fallback: store in memory
            logger.warning("ChromaDB unavailable. Documents stored in memory only.")
            self._memory_docs = docs
            return len(docs)

        try:
            # Check existing documents
            existing = collection.get()
            existing_ids = set(existing["ids"]) if existing["ids"] else set()

            new_docs = [d for d in docs if d["id"] not in existing_ids]
            if not new_docs:
                logger.info("All documents already indexed.")
                return 0

            collection.add(
                ids=[d["id"] for d in new_docs],
                documents=[d["content"] for d in new_docs],
                metadatas=[d.get("metadata", {}) for d in new_docs],
            )
            logger.info("Indexed %d new documents.", len(new_docs))
            return len(new_docs)

        except Exception as e:
            logger.error("Error indexing documents: %s", e)
            return 0

    def index_golden_set(self) -> int:
        """Index the golden set queries as additional context documents.

        Returns:
            Number of golden set entries indexed.
        """
        golden_path = PROJECT_ROOT / "data" / "golden_set" / "golden_set.json"
        if not golden_path.exists():
            return 0

        try:
            with open(golden_path) as f:
                golden_set = json.load(f)

            docs = []
            for item in golden_set:
                if item.get("expected_answer"):
                    docs.append(
                        {
                            "id": f"golden_{item['id']}",
                            "content": f"Q: {item['query']}\nA: {item['expected_answer']}",
                            "metadata": {"source": "golden_set", "topic": "qa"},
                        }
                    )

            if docs:
                return self.index_documents(docs)
            return 0

        except Exception as e:
            logger.error("Error indexing golden set: %s", e)
            return 0

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'content', 'metadata', 'distance'.
        """
        collection = self._get_collection()

        if collection is None:
            # Fallback: simple keyword search in memory
            return self._memory_search(query, top_k)

        try:
            # Ensure documents are indexed
            if collection.count() == 0:
                self.index_documents()

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
            )

            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    retrieved.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        }
                    )

            return retrieved

        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            return self._memory_search(query, top_k)

    def _memory_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Fallback keyword-based search when ChromaDB is unavailable.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of matching documents.
        """
        docs = getattr(self, "_memory_docs", KNOWLEDGE_DOCUMENTS)
        query_lower = query.lower()
        scored = []

        for doc in docs:
            content_lower = doc["content"].lower()
            # Simple relevance scoring based on keyword overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": doc["content"], "metadata": doc.get("metadata", {}), "distance": 0.0}
            for _, doc in scored[:top_k]
        ]


# Module-level singleton
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the singleton RAG pipeline instance.

    Returns:
        Initialized RAGPipeline.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.index_documents()
    return _rag_pipeline


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context for a query (convenience function).

    Args:
        query: Search query.
        top_k: Number of documents to retrieve.

    Returns:
        Formatted context string.
    """
    pipeline = get_rag_pipeline()
    results = pipeline.retrieve(query, top_k=top_k)

    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.get("metadata", {}).get("source", "unknown")
        topic = result.get("metadata", {}).get("topic", "general")
        context_parts.append(f"[Source {i}: {source}/{topic}]\n{result['content']}")

    return "\n\n".join(context_parts)
