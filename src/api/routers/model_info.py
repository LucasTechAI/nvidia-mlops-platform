"""
Model Information Router.

Provides endpoints for model architecture, training history, and HPO results.
"""

import logging
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _find_checkpoint() -> Path | None:
    """Find the best model checkpoint."""
    candidates = [
        PROJECT_ROOT / "models" / "best_model.pth",
        PROJECT_ROOT / "models" / "best_model.pt",
        PROJECT_ROOT / "data" / "models" / "checkpoints" / "best_model.pt",
        PROJECT_ROOT / "data" / "models" / "checkpoints" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_checkpoint() -> dict | None:
    """Load and normalize checkpoint data."""
    path = _find_checkpoint()
    if path is None:
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        # Handle bare state_dict
        if (
            isinstance(data, dict)
            and "model_state_dict" not in data
            and all(isinstance(v, torch.Tensor) for v in list(data.values())[:3])
        ):
            state_dict = data
            input_size = state_dict["lstm.weight_ih_l0"].shape[1] if "lstm.weight_ih_l0" in state_dict else 5
            hidden_size = state_dict["lstm.weight_hh_l0"].shape[1] if "lstm.weight_hh_l0" in state_dict else 128
            output_size = state_dict["fc.bias"].shape[0] if "fc.bias" in state_dict else 1
            num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l")) or 2
            data = {
                "model_state_dict": state_dict,
                "model_config": {
                    "input_size": int(input_size),
                    "hidden_size": int(hidden_size),
                    "output_size": int(output_size),
                    "num_layers": int(num_layers),
                },
                "epoch": 0,
                "loss": 0.0,
            }
        return data
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None


def _count_parameters(model_state_dict: dict) -> dict:
    """Count model parameters by layer."""
    layers = {}
    total = 0
    for name, param in model_state_dict.items():
        count = int(param.numel())
        layers[name] = {
            "shape": [int(s) for s in param.shape],
            "count": count,
            "dtype": str(param.dtype),
        }
        total += count
    return {"layers": layers, "total": total, "trainable": total}


@router.get("/info")
async def get_model_info():
    """Get model architecture, configuration, and parameter analysis."""
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found")

    config = checkpoint.get("model_config", {})
    state = checkpoint.get("model_state_dict", {})
    training_info = checkpoint.get("training_info", {})
    # Support both key names: "test_results" (actual) and "test_metrics" (legacy)
    test_metrics = checkpoint.get("test_results", checkpoint.get("test_metrics", {}))

    # Serialize config values
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config_serializable[k] = v.item() if v.numel() == 1 else v.tolist()
        else:
            config_serializable[k] = v

    # Parameter analysis
    params = _count_parameters(state) if state else {"layers": {}, "total": 0, "trainable": 0}

    # Serialize test_metrics
    metrics_serializable = {}
    for k, v in test_metrics.items():
        if isinstance(v, (torch.Tensor,)):
            metrics_serializable[k] = float(v.item()) if v.numel() == 1 else v.tolist()
        elif isinstance(v, float):
            metrics_serializable[k] = round(v, 6)
        else:
            metrics_serializable[k] = v

    # Training info serialization
    training_serializable = {}
    for k, v in training_info.items():
        if isinstance(v, torch.Tensor):
            training_serializable[k] = float(v.item()) if v.numel() == 1 else v.tolist()
        elif isinstance(v, float):
            training_serializable[k] = round(v, 6)
        else:
            training_serializable[k] = v

    # Extract epoch/loss from training_info if not at top level
    best_epoch = checkpoint.get("best_epoch") or training_info.get("Best Epoch", 0)
    total_epochs = checkpoint.get("epoch") or training_info.get("Total Epochs", 0)
    best_loss = checkpoint.get("best_loss") or training_info.get("Best Val Loss")
    loss = checkpoint.get("loss") or (best_loss if best_loss is not None else 0.0)

    return {
        "model_config": config_serializable,
        "parameters": params,
        "training_info": training_serializable,
        "test_metrics": metrics_serializable,
        "epoch": int(total_epochs),
        "best_epoch": int(best_epoch),
        "loss": float(loss),
        "best_loss": float(best_loss) if best_loss is not None else None,
        "features": checkpoint.get("features", []),
    }


@router.get("/training-history")
async def get_training_history():
    """Get training curves (loss, RMSE, MAE, R² per epoch) + test metrics."""
    import math

    checkpoint = _load_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found")

    # Support both formats: nested "training_history" dict or top-level lists
    history = checkpoint.get("training_history", {})
    if not history:
        # Try top-level train_losses / val_losses (legacy checkpoint format)
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        if train_losses or val_losses:
            history = {}
            if train_losses:
                history["train_loss"] = train_losses
                # Derive train_rmse from MSE loss (rmse = sqrt(mse))
                history["train_rmse"] = [math.sqrt(v) for v in train_losses]
            if val_losses:
                history["val_loss"] = val_losses
                # Derive val_rmse from MSE loss
                history["val_rmse"] = [math.sqrt(v) for v in val_losses]

    if not history:
        raise HTTPException(status_code=404, detail="No training history in checkpoint")

    # Convert numpy/tensor arrays and scalars to plain Python floats
    result = {}
    for key, values in history.items():
        if hasattr(values, "tolist"):
            result[key] = values.tolist()
        elif isinstance(values, list):
            # Skip empty lists (e.g. test_* when test_data was not provided during training)
            if len(values) == 0:
                continue
            result[key] = [float(v) for v in values]
        else:
            result[key] = values

    # If per-epoch test metrics already exist in history, skip re-computation
    has_per_epoch_test = (
        isinstance(result.get("test_loss"), list)
        and len(result.get("test_loss", [])) > 1
    )

    if not has_per_epoch_test:
        # ── Compute test metrics in *normalized* space (single-value fallback) ──
        # Train/val curves use normalized data so test must too for comparison.
        try:
            import sqlite3

            import numpy as np
            import torch

            from src.api.dependencies import ModelState
            from src.config import DATABASE_PATH
            from src.data.preprocessing import create_sequences

            state = ModelState()
            if state.is_ready and state.model is not None and state.scaler is not None:
                conn = sqlite3.connect(DATABASE_PATH)
                df = __import__("pandas").read_sql(
                    "SELECT * FROM nvidia_stock ORDER BY date",
                    conn,
                )
                conn.close()
                df.columns = [c.capitalize() for c in df.columns]

                feature_cols = ["Open", "High", "Low", "Close", "Volume"]
                available = [c for c in feature_cols if c in df.columns]
                raw = df[available].values
                normalized = state.scaler.transform(raw)

                seq_len = state.model_config.get("sequence_length", 60)
                X, y = create_sequences(normalized, seq_len)

                # Test split = last 15 %
                n = len(X)
                test_start = int(n * 0.85)
                X_test = torch.FloatTensor(X[test_start:]).to(state.device)
                y_test = torch.FloatTensor(y[test_start:]).to(state.device)

                state.model.eval()
                with torch.no_grad():
                    preds = state.model(X_test)
                preds_np = preds.cpu().numpy().flatten()
                y_np = y_test.cpu().numpy().flatten()

                test_mse = float(np.mean((preds_np - y_np) ** 2))
                test_rmse = float(np.sqrt(test_mse))
                test_mae = float(np.mean(np.abs(preds_np - y_np)))
                ss_res = float(np.sum((y_np - preds_np) ** 2))
                ss_tot = float(np.sum((y_np - np.mean(y_np)) ** 2))
                test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                result["test_loss"] = test_mse
                result["test_rmse"] = test_rmse
                result["test_mae"] = test_mae
                result["test_r2"] = float(test_r2)
                result["test_n_samples"] = len(y_np)
        except Exception as e:
            logger.warning("Could not compute normalized test metrics: %s", e)

    return result


@router.get("/hpo-results")
async def get_hpo_results():
    """Get hyperparameter optimization results."""
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found")

    # Try checkpoint first
    hpo = checkpoint.get("hpo_best_params", {})
    if hpo:
        return {"source": "checkpoint", "best_params": hpo}

    # Try MLflow
    try:
        from src.dashboard.components.metrics import load_hpo_results

        hpo_data = load_hpo_results()
        if hpo_data:
            return {"source": "mlflow", "best_params": hpo_data}
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="No HPO results found")
