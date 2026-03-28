"""Champion-Challenger model evaluation pipeline.

Implements automated model promotion with champion-challenger comparison:
1. Detect drift → trigger retraining
2. Train challenger model on new data
3. Compare challenger vs champion on holdout set
4. Only promote if challenger significantly outperforms champion

Thresholds:
    - δ RMSE ≤ -0.5% → promote challenger as new champion
    - δ RMSE > 0      → keep champion (challenger is worse)

References:
    - MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
    - Sato, Wider, Windheuser (2019) — Continuous Delivery for ML
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "champion_challenger"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Promotion threshold: challenger must improve RMSE by at least this fraction
IMPROVEMENT_THRESHOLD = 0.005  # 0.5%


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "directional_accuracy": self.directional_accuracy,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonResult:
    """Result of a champion-challenger comparison."""

    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    promote: bool = False
    reason: str = ""
    rmse_delta: float = 0.0
    rmse_delta_pct: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "champion": self.champion_metrics.to_dict(),
            "challenger": self.challenger_metrics.to_dict(),
            "promote": self.promote,
            "reason": self.reason,
            "rmse_delta": round(self.rmse_delta, 6),
            "rmse_delta_pct": round(self.rmse_delta_pct, 4),
            "timestamp": self.timestamp,
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: Any,
    device: str = "cpu",
    target_idx: int = 0,
) -> ModelMetrics:
    """Evaluate a model on a dataset and return comprehensive metrics.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the evaluation dataset.
        scaler: Fitted scaler for inverse transformation.
        device: Device for inference.
        target_idx: Index of the target column for inverse transform.

    Returns:
        ModelMetrics with RMSE, MAE, MAPE, R², directional accuracy.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)

            # Extract the target feature prediction
            if preds.dim() == 2 and preds.shape[1] > 1:
                preds = preds[:, target_idx : target_idx + 1]

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Inverse transform if scaler available
    if scaler is not None:
        n_features = getattr(scaler, "n_features_in_", 1)
        if n_features > 1:
            dummy_pred = np.zeros((len(predictions), n_features))
            dummy_pred[:, target_idx] = predictions
            predictions = scaler.inverse_transform(dummy_pred)[:, target_idx]

            dummy_tgt = np.zeros((len(targets), n_features))
            dummy_tgt[:, target_idx] = targets
            targets = scaler.inverse_transform(dummy_tgt)[:, target_idx]
        else:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Compute metrics
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mae = float(np.mean(np.abs(predictions - targets)))

    # MAPE (avoid division by zero)
    mask = np.abs(targets) > 1e-8
    mape = float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100) if mask.any() else 0.0

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy
    if len(targets) > 1:
        actual_dir = np.sign(np.diff(targets))
        pred_dir = np.sign(np.diff(predictions))
        dir_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_accuracy = 0.0

    return ModelMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        directional_accuracy=dir_accuracy,
    )


def compare_models(
    champion_metrics: ModelMetrics,
    challenger_metrics: ModelMetrics,
    threshold: float = IMPROVEMENT_THRESHOLD,
) -> ComparisonResult:
    """Compare champion and challenger models.

    Promotion criteria:
        Challenger is promoted if its RMSE is at least `threshold` (0.5%)
        better than the champion.

    Args:
        champion_metrics: Metrics of the current production model.
        challenger_metrics: Metrics of the newly trained model.
        threshold: Minimum relative improvement to promote.

    Returns:
        ComparisonResult with promotion decision and reasoning.
    """
    rmse_delta = challenger_metrics.rmse - champion_metrics.rmse
    rmse_delta_pct = rmse_delta / champion_metrics.rmse if champion_metrics.rmse > 0 else 0.0

    result = ComparisonResult(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        rmse_delta=rmse_delta,
        rmse_delta_pct=rmse_delta_pct,
    )

    if rmse_delta_pct <= -threshold:
        result.promote = True
        result.reason = (
            f"Challenger RMSE ({challenger_metrics.rmse:.4f}) is "
            f"{abs(rmse_delta_pct)*100:.2f}% better than champion "
            f"({champion_metrics.rmse:.4f}). Promoting."
        )
        logger.info("✅ PROMOTE: %s", result.reason)
    elif rmse_delta < 0:
        result.promote = False
        result.reason = (
            f"Challenger RMSE improved by {abs(rmse_delta_pct)*100:.2f}% "
            f"but below threshold ({threshold*100:.1f}%). Keeping champion."
        )
        logger.info("⚠️ NO PROMOTE (below threshold): %s", result.reason)
    else:
        result.promote = False
        result.reason = (
            f"Challenger RMSE ({challenger_metrics.rmse:.4f}) is worse "
            f"than champion ({champion_metrics.rmse:.4f}). Keeping champion."
        )
        logger.info("❌ NO PROMOTE: %s", result.reason)

    return result


def run_champion_challenger(
    champion_path: Optional[str] = None,
    experiment_name: str = "champion_challenger",
    retrain_on_drift: bool = True,
) -> dict:
    """Run the full champion-challenger pipeline.

    Steps:
        1. Check for drift
        2. If drift detected (or forced), train challenger
        3. Evaluate both on holdout set
        4. Compare and decide promotion
        5. Log results to MLflow

    Args:
        champion_path: Path to champion model checkpoint. Defaults to best_model.pth.
        experiment_name: MLflow experiment name.
        retrain_on_drift: If True, only retrain when drift detected.

    Returns:
        Dictionary with pipeline results.
    """
    from src.monitoring.drift import detect_drift_from_db
    from src.training.train import train_model

    result = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": False,
        "retrained": False,
        "comparison": None,
        "promoted": False,
    }

    # Step 1: Drift detection
    logger.info("=" * 60)
    logger.info("Step 1: Drift Detection")
    logger.info("=" * 60)

    drift_results = detect_drift_from_db()
    drift_detected = drift_results.get("drift_detected", False)
    result["drift_detected"] = drift_detected
    result["drift_results"] = drift_results

    if retrain_on_drift and not drift_detected:
        logger.info("No drift detected. Skipping retraining.")
        result["reason"] = "No drift detected"
        _save_result(result)
        return result

    # Step 2: Train challenger
    logger.info("=" * 60)
    logger.info("Step 2: Training Challenger Model")
    logger.info("=" * 60)

    try:
        training_result = train_model(experiment_name=f"{experiment_name}_challenger")
        result["retrained"] = True
        result["training_result"] = {
            "run_id": training_result.get("run_id"),
            "best_val_loss": training_result.get("best_val_loss"),
        }
    except Exception as e:
        logger.error("Challenger training failed: %s", str(e))
        result["error"] = str(e)
        _save_result(result)
        return result

    # Step 3 & 4: Compare (metrics already logged by train_model)
    logger.info("=" * 60)
    logger.info("Step 3: Champion-Challenger Comparison")
    logger.info("=" * 60)

    champion_loss = training_result.get("champion_val_loss", float("inf"))
    challenger_loss = training_result.get("best_val_loss", float("inf"))

    champion_m = ModelMetrics(rmse=champion_loss)
    challenger_m = ModelMetrics(rmse=challenger_loss)
    comparison = compare_models(champion_m, challenger_m)
    result["comparison"] = comparison.to_dict()
    result["promoted"] = comparison.promote

    # Step 5: Log to MLflow
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="champion_challenger_evaluation"):
            mlflow.log_params(
                {
                    "drift_detected": drift_detected,
                    "retrained": result["retrained"],
                    "promoted": comparison.promote,
                }
            )
            mlflow.log_metrics(
                {
                    "champion_rmse": champion_m.rmse,
                    "challenger_rmse": challenger_m.rmse,
                    "rmse_delta": comparison.rmse_delta,
                    "rmse_delta_pct": comparison.rmse_delta_pct,
                }
            )
            mlflow.set_tag("pipeline", "champion_challenger")
            mlflow.set_tag("promotion_decision", "promote" if comparison.promote else "keep_champion")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", str(e))

    _save_result(result)
    return result


def _save_result(result: dict) -> None:
    """Save pipeline result to JSON."""
    output_path = RESULTS_DIR / "latest_comparison.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)
