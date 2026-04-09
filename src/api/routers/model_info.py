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
    test_metrics = checkpoint.get("test_metrics", {})

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

    return {
        "model_config": config_serializable,
        "parameters": params,
        "training_info": training_serializable,
        "test_metrics": metrics_serializable,
        "epoch": checkpoint.get("epoch", 0),
        "best_epoch": checkpoint.get("best_epoch", 0),
        "loss": float(checkpoint.get("loss", 0.0)),
        "best_loss": float(checkpoint.get("best_loss", 0.0)) if checkpoint.get("best_loss") else None,
        "features": checkpoint.get("features", []),
    }


@router.get("/training-history")
async def get_training_history():
    """Get training curves (loss, RMSE, MAE, R² per epoch)."""
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found")

    history = checkpoint.get("training_history", {})
    if not history:
        raise HTTPException(status_code=404, detail="No training history in checkpoint")

    # Convert numpy/tensor arrays to lists
    result = {}
    for key, values in history.items():
        if hasattr(values, "tolist"):
            result[key] = values.tolist()
        elif isinstance(values, list):
            result[key] = [float(v) if isinstance(v, (int, float)) else v for v in values]
        else:
            result[key] = values

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
