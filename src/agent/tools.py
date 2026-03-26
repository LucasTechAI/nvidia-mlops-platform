"""Custom tools for the financial domain ReAct agent.

Tools:
    1. Historical data query — query SQLite NVIDIA prices
    2. Prediction — invoke trained LSTM model
    3. Metrics analysis — fetch MLflow metrics
    4. RAG search — retrieve relevant context from documents
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Tool 1: Historical Data Query
# =============================================================================


def query_stock_data(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 30,
) -> str:
    """Query historical NVIDIA stock data from SQLite database.

    Interprets natural-language queries about stock prices and returns
    formatted results from the database.

    Args:
        query: Natural language query about stock data.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        limit: Maximum number of rows to return.

    Returns:
        Formatted string with stock data results.
    """
    db_path = PROJECT_ROOT / "data" / "nvidia_stock.db"
    if not db_path.exists():
        return "Error: Stock database not found. Run ETL pipeline first."

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Build query based on natural language input
        query_lower = query.lower()

        if any(word in query_lower for word in ["último", "última", "recente", "ontem", "latest", "recent"]):
            sql = "SELECT * FROM nvidia_stock ORDER BY date DESC LIMIT ?"
            params: tuple = (min(limit, 10),)
        elif any(word in query_lower for word in ["máximo", "máxima", "maior", "highest", "max"]):
            sql = "SELECT * FROM nvidia_stock ORDER BY close DESC LIMIT ?"
            params = (5,)
        elif any(word in query_lower for word in ["mínimo", "mínima", "menor", "lowest", "min"]):
            sql = "SELECT * FROM nvidia_stock ORDER BY close ASC LIMIT ?"
            params = (5,)
        elif any(word in query_lower for word in ["média", "average", "mean"]):
            sql = """
                SELECT
                    ROUND(AVG(open), 2) as avg_open,
                    ROUND(AVG(high), 2) as avg_high,
                    ROUND(AVG(low), 2) as avg_low,
                    ROUND(AVG(close), 2) as avg_close,
                    ROUND(AVG(volume), 0) as avg_volume,
                    COUNT(*) as total_days
                FROM nvidia_stock
            """
            params = ()
        elif any(word in query_lower for word in ["volume", "negociação"]):
            sql = "SELECT * FROM nvidia_stock ORDER BY volume DESC LIMIT ?"
            params = (5,)
        elif start_date and end_date:
            sql = "SELECT * FROM nvidia_stock WHERE date BETWEEN ? AND ? ORDER BY date DESC LIMIT ?"
            params = (start_date, end_date, limit)
        else:
            sql = "SELECT * FROM nvidia_stock ORDER BY date DESC LIMIT ?"
            params = (limit,)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No data found for the given query."

        # Format results
        result_lines: list[str] = []
        for row in rows:
            row_dict = dict(row)
            if "date" in row_dict:
                result_lines.append(
                    f"Date: {row_dict['date']} | "
                    f"Open: ${float(row_dict.get('open', 0)):.2f} | "
                    f"High: ${float(row_dict.get('high', 0)):.2f} | "
                    f"Low: ${float(row_dict.get('low', 0)):.2f} | "
                    f"Close: ${float(row_dict.get('close', 0)):.2f} | "
                    f"Volume: {float(row_dict.get('volume', 0)):,.0f}"
                )
            else:
                result_lines.append(" | ".join(f"{k}: {v}" for k, v in row_dict.items()))

        return f"NVIDIA Stock Data ({len(rows)} rows):\n" + "\n".join(result_lines)

    except Exception as e:
        logger.error("Error querying stock data: %s", e)
        return f"Error querying database: {e}"


# =============================================================================
# Tool 2: LSTM Prediction
# =============================================================================


def predict_stock_prices(horizon: str = "5") -> str:
    """Generate NVIDIA stock price predictions using the trained LSTM model.

    Loads the latest model checkpoint and scaler, generates predictions
    for the specified horizon, and returns formatted results.

    Args:
        horizon: Number of days to forecast (1-30), as string.

    Returns:
        Formatted string with predicted prices.
    """
    try:
        n_days = int(horizon)
    except (ValueError, TypeError):
        n_days = 5
    n_days = max(1, min(n_days, 30))

    model_path = PROJECT_ROOT / "models" / "best_model.pth"
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"

    if not model_path.exists():
        return "Error: No trained model found. Run training pipeline first."
    if not scaler_path.exists():
        return "Error: No scaler found. Run training pipeline first."

    try:
        import pickle

        from src.models.lstm_model import NvidiaLSTM

        # Load scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model_config = checkpoint.get("model_config", {})
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            model_config = {}

        model = NvidiaLSTM(
            input_size=model_config.get("input_size", 5),
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
            output_size=model_config.get("output_size", 5),
            dropout=model_config.get("dropout", 0.2),
            bidirectional=model_config.get("bidirectional", False),
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Load recent data for the initial sequence
        db_path = PROJECT_ROOT / "data" / "nvidia_stock.db"
        if not db_path.exists():
            return "Error: Stock database not found."

        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query("SELECT * FROM nvidia_stock ORDER BY date DESC LIMIT 60", conn)
        conn.close()
        df = df.sort_values("date").reset_index(drop=True)

        # Determine feature columns (handle both lower and capitalized)
        feature_cols_lower = ["open", "high", "low", "close", "volume"]
        feature_cols_upper = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in feature_cols_upper if c in df.columns]
        if len(available) != scaler.n_features_in_:
            available = [c for c in feature_cols_lower if c in df.columns]

        close_col = "Close" if "Close" in available else "close"
        close_idx = available.index(close_col)
        features = df[available].values[-60:]

        # Normalize and predict
        features_norm = scaler.transform(features)
        sequence = torch.FloatTensor(features_norm).unsqueeze(0)

        predictions = []
        current_seq = sequence.clone()
        with torch.no_grad():
            for _ in range(n_days):
                pred = model(current_seq)
                predictions.append(pred.cpu().numpy().flatten())
                new_input = pred.view(1, 1, -1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_input], dim=1)

        preds_array = np.array(predictions)
        preds_orig = scaler.inverse_transform(preds_array)
        pred_prices = preds_orig[:, close_idx]

        last_price = float(df[close_col].iloc[-1])
        last_date = df["date"].iloc[-1]

        result_lines = [f"NVIDIA ({last_date}) Last Close: ${last_price:.2f}"]
        result_lines.append(f"Forecast for next {n_days} days:")
        for i, price in enumerate(pred_prices, 1):
            change = ((price - last_price) / last_price) * 100
            arrow = "📈" if change > 0 else "📉"
            result_lines.append(f"  Day {i}: ${price:.2f} ({change:+.2f}%) {arrow}")

        avg_pred = float(np.mean(pred_prices))
        total_change = ((pred_prices[-1] - last_price) / last_price) * 100
        result_lines.append(f"\nAverage predicted: ${avg_pred:.2f}")
        result_lines.append(f"Total change: {total_change:+.2f}%")

        return "\n".join(result_lines)

    except Exception as e:
        logger.error("Error generating predictions: %s", e)
        return f"Error generating predictions: {e}"


# =============================================================================
# Tool 3: Model Metrics Analysis
# =============================================================================


def get_model_metrics(query: str = "") -> str:
    """Fetch and analyze model performance metrics from checkpoint.

    Returns training info, test metrics, and model configuration.

    Args:
        query: Optional query to filter which metrics to return.

    Returns:
        Formatted string with model metrics.
    """
    model_path = PROJECT_ROOT / "models" / "best_model.pth"

    if not model_path.exists():
        return "Error: No model checkpoint found."

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        result_lines = ["📊 NVIDIA LSTM Model Metrics\n"]

        # Training info
        training_info = checkpoint.get("training_info", {})
        if training_info:
            result_lines.append("🎯 Training Overview:")
            for key, value in training_info.items():
                if isinstance(value, float):
                    result_lines.append(f"  {key}: {value:.6f}")
                else:
                    result_lines.append(f"  {key}: {value}")

        # Test results
        test_results = checkpoint.get("test_results", {})
        if test_results:
            result_lines.append("\n📈 Test Set Performance:")
            metric_formats = {
                "rmse": ("RMSE", "${:.4f}"),
                "mae": ("MAE", "${:.4f}"),
                "mape": ("MAPE", "{:.2f}%"),
                "r2_score": ("R² Score", "{:.4f}"),
                "correlation": ("Correlation", "{:.4f}"),
                "directional_accuracy": ("Dir. Accuracy", "{:.1f}%"),
                "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
                "max_drawdown": ("Max Drawdown", "{:.1f}%"),
            }
            for key, (label, fmt) in metric_formats.items():
                value = test_results.get(key)
                if value is not None and value != 0:
                    result_lines.append(f"  {label}: {fmt.format(value)}")

        # Model config
        model_config = checkpoint.get("model_config", {})
        if model_config:
            result_lines.append("\n🧠 Model Architecture:")
            for key, value in model_config.items():
                result_lines.append(f"  {key}: {value}")

        if not training_info and not test_results:
            result_lines.append("No detailed metrics available in checkpoint.")

        return "\n".join(result_lines)

    except Exception as e:
        logger.error("Error loading model metrics: %s", e)
        return f"Error loading metrics: {e}"


# =============================================================================
# Tool 4: RAG Document Search
# =============================================================================


def search_documents(query: str) -> str:
    """Search project documents and knowledge base using RAG pipeline.

    Retrieves relevant context from indexed documents about NVIDIA,
    the model, and financial analysis.

    Args:
        query: Search query in natural language.

    Returns:
        Relevant context retrieved from the document store.
    """
    try:
        from src.agent.rag_pipeline import retrieve_context

        results = retrieve_context(query, top_k=3)
        if not results:
            return "No relevant documents found for the query."
        return results
    except Exception as e:
        logger.error("Error searching documents: %s", e)
        return f"Error searching documents: {e}"


# =============================================================================
# Tool registry for the agent
# =============================================================================

TOOL_REGISTRY = {
    "query_stock_data": {
        "function": query_stock_data,
        "name": "query_stock_data",
        "description": (
            "Query historical NVIDIA stock price data from the database. "
            "Use this tool to get real stock prices, trading volumes, "
            "historical highs/lows, averages, and recent closing prices. "
            "Input should be a natural language query about NVIDIA stock data."
        ),
    },
    "predict_stock_prices": {
        "function": predict_stock_prices,
        "name": "predict_stock_prices",
        "description": (
            "Generate NVIDIA stock price predictions using the trained LSTM model. "
            "Returns predicted closing prices for the next N days (1-30). "
            "Input should be the number of days to forecast as a string integer."
        ),
    },
    "get_model_metrics": {
        "function": get_model_metrics,
        "name": "get_model_metrics",
        "description": (
            "Get performance metrics of the trained NVIDIA LSTM model. "
            "Returns training history, test set metrics (RMSE, MAE, MAPE, R², "
            "correlation, Sharpe ratio), and model architecture details. "
            "Use this to evaluate model quality and reliability."
        ),
    },
    "search_documents": {
        "function": search_documents,
        "name": "search_documents",
        "description": (
            "Search the knowledge base for information about NVIDIA, "
            "stock market analysis, model methodology, and financial concepts. "
            "Use this for general questions that need context beyond raw data."
        ),
    },
}
