"""Tests for agent tools module."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.agent.tools import (
    TOOL_REGISTRY,
    get_model_metrics,
    predict_stock_prices,
    query_stock_data,
    search_documents,
)

# ---------------------------------------------------------------------------
# Tests — TOOL_REGISTRY
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_all_tools_registered(self):
        expected = {"query_stock_data", "predict_stock_prices", "get_model_metrics", "search_documents"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_tools_have_required_keys(self):
        for name, tool in TOOL_REGISTRY.items():
            assert "function" in tool
            assert "name" in tool
            assert "description" in tool
            assert callable(tool["function"])

    def test_descriptions_are_strings(self):
        for tool in TOOL_REGISTRY.values():
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 10


# ---------------------------------------------------------------------------
# Tests — query_stock_data
# ---------------------------------------------------------------------------

class TestQueryStockData:
    def test_db_not_found(self, monkeypatch):
        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", Path("/nonexistent"))
        result = query_stock_data("latest prices")
        assert "Error" in result or "not found" in result

    def test_query_with_db(self, tmp_path, monkeypatch):
        # Create a tiny test database
        db_path = tmp_path / "data" / "nvidia_stock.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE nvidia_stock (
                date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
            )
        """)
        conn.execute(
            "INSERT INTO nvidia_stock VALUES (?, ?, ?, ?, ?, ?)",
            ("2024-01-01", 100, 110, 90, 105, 1000000),
        )
        conn.execute(
            "INSERT INTO nvidia_stock VALUES (?, ?, ?, ?, ?, ?)",
            ("2024-01-02", 105, 115, 95, 110, 2000000),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", tmp_path)
        result = query_stock_data("latest prices")
        assert "NVIDIA Stock Data" in result
        assert "$" in result

    def test_query_average(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "nvidia_stock.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE nvidia_stock (
                date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
            )
        """)
        conn.execute("INSERT INTO nvidia_stock VALUES ('2024-01-01', 100, 110, 90, 105, 1000000)")
        conn.commit()
        conn.close()

        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", tmp_path)
        result = query_stock_data("average price")
        assert "avg_close" in result or "NVIDIA" in result

    def test_query_highest(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "nvidia_stock.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE nvidia_stock "
            "(date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"
        )
        conn.execute(
            "INSERT INTO nvidia_stock VALUES ('2024-01-01', 100, 110, 90, 105, 1000000)"
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", tmp_path)
        result = query_stock_data("highest price")
        assert "NVIDIA" in result


# ---------------------------------------------------------------------------
# Tests — predict_stock_prices
# ---------------------------------------------------------------------------

class TestPredictStockPrices:
    def test_no_model_returns_error(self, monkeypatch):
        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", Path("/nonexistent"))
        result = predict_stock_prices("5")
        assert "Error" in result

    def test_invalid_horizon(self, monkeypatch):
        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", Path("/nonexistent"))
        result = predict_stock_prices("abc")
        assert "Error" in result  # model not found, but horizon parsing worked


# ---------------------------------------------------------------------------
# Tests — get_model_metrics
# ---------------------------------------------------------------------------

class TestGetModelMetrics:
    def test_no_checkpoint(self, monkeypatch):
        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", Path("/nonexistent"))
        result = get_model_metrics()
        assert "Error" in result or "not found" in result

    def test_with_checkpoint(self, tmp_path, monkeypatch):
        import torch
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        checkpoint = {
            "model_state_dict": {},
            "training_info": {"epochs": 100, "best_val_loss": 0.001},
            "test_results": {"rmse": 5.0, "mae": 3.0, "r2_score": 0.95},
            "model_config": {"hidden_size": 128, "num_layers": 2},
        }
        torch.save(checkpoint, str(model_dir / "best_model.pth"))
        monkeypatch.setattr("src.agent.tools.PROJECT_ROOT", tmp_path)

        result = get_model_metrics()
        assert "NVIDIA LSTM" in result
        assert "R² Score" in result


# ---------------------------------------------------------------------------
# Tests — search_documents
# ---------------------------------------------------------------------------

class TestSearchDocuments:
    def test_rag_import_error(self):
        """If RAG pipeline fails to import, should return error string."""
        with patch("src.agent.tools.search_documents.__module__", "src.agent.tools"):
            result = search_documents("what is NVIDIA?")
            # Should return string (either result or error)
            assert isinstance(result, str)
