"""Feature importance analysis for the NVIDIA LSTM model.

Implements permutation-based feature importance to explain which input
features (Open, High, Low, Close, Volume) contribute most to predictions.

Permutation importance is model-agnostic: for each feature, we shuffle
its values across samples and measure the increase in prediction error.
A large increase indicates high importance.

References:
    - Breiman, "Random Forests", Machine Learning, 2001
    - Molnar, "Interpretable Machine Learning", 2022
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "explainability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default feature names matching the NVIDIA stock dataset
DEFAULT_FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]


def _compute_rmse(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    """Compute RMSE for a given dataset.

    Args:
        model: Trained PyTorch model.
        X: Input features (n_samples, seq_len, n_features).
        y: Target values (n_samples, output_size).
        device: Torch device.

    Returns:
        Root Mean Squared Error.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def compute_permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict:
    """Compute permutation feature importance for the LSTM model.

    For each feature, shuffles its values across all samples and
    sequence positions, then measures the increase in RMSE.

    Args:
        model: Trained PyTorch model.
        X: Input array of shape (n_samples, sequence_length, n_features).
        y: Target array of shape (n_samples, output_size).
        feature_names: Optional list of feature names. Defaults to DEFAULT_FEATURE_NAMES.
        n_repeats: Number of shuffle repetitions per feature.
        device: Torch device (defaults to CPU).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - feature_names: List of feature names.
            - importances_mean: Mean importance per feature.
            - importances_std: Std of importance per feature.
            - baseline_rmse: RMSE without permutation.
            - details: Per-feature, per-repeat scores.
    """
    if device is None:
        device = torch.device("cpu")

    rng = np.random.RandomState(random_state)

    n_features = X.shape[2]
    if feature_names is None:
        if n_features <= len(DEFAULT_FEATURE_NAMES):
            feature_names = DEFAULT_FEATURE_NAMES[:n_features]
        else:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Baseline RMSE (no permutation)
    baseline_rmse = _compute_rmse(model, X, y, device)
    logger.info(f"Baseline RMSE: {baseline_rmse:.6f}")

    importances = {name: [] for name in feature_names}

    for feat_idx, feat_name in enumerate(feature_names):
        for repeat in range(n_repeats):
            # Copy and shuffle the feature column across samples
            X_permuted = X.copy()
            shuffled_idx = rng.permutation(X.shape[0])
            X_permuted[:, :, feat_idx] = X[shuffled_idx, :, feat_idx]

            # Compute RMSE with permuted feature
            permuted_rmse = _compute_rmse(model, X_permuted, y, device)

            # Importance = increase in error
            importance = permuted_rmse - baseline_rmse
            importances[feat_name].append(importance)

        mean_imp = np.mean(importances[feat_name])
        logger.info(f"  {feat_name}: importance = {mean_imp:.6f}")

    # Aggregate results
    importances_mean = [float(np.mean(importances[name])) for name in feature_names]
    importances_std = [float(np.std(importances[name])) for name in feature_names]

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "permutation_importance",
        "baseline_rmse": float(baseline_rmse),
        "n_repeats": n_repeats,
        "n_samples": int(X.shape[0]),
        "sequence_length": int(X.shape[1]),
        "feature_names": feature_names,
        "importances_mean": importances_mean,
        "importances_std": importances_std,
        "details": {name: [float(v) for v in importances[name]] for name in feature_names},
    }

    # Save results to JSON
    output_path = RESULTS_DIR / "permutation_importance.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved importance results to {output_path}")

    return results


def plot_feature_importance(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> str:
    """Generate a horizontal bar chart of feature importances.

    Args:
        results: Output from compute_permutation_importance.
        save_path: Optional path to save the plot. Defaults to RESULTS_DIR.
        figsize: Figure size.

    Returns:
        Path to the saved plot image.
    """
    feature_names = results["feature_names"]
    importances_mean = np.array(results["importances_mean"])
    importances_std = np.array(results["importances_std"])

    # Sort by importance
    sorted_idx = np.argsort(importances_mean)
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    importances_sorted = importances_mean[sorted_idx]
    std_sorted = importances_std[sorted_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_names)))
    bars = ax.barh(
        range(len(feature_names)),
        importances_sorted,
        xerr=std_sorted,
        color=[colors[i] for i in range(len(feature_names))],
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
    )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names_sorted, fontsize=11)
    ax.set_xlabel("Mean RMSE Increase (Permutation Importance)", fontsize=12)
    ax.set_title(
        "NVIDIA LSTM — Feature Importance (Permutation Method)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Add value labels on bars
    for bar, val in zip(bars, importances_sorted):
        ax.text(
            bar.get_width() + max(std_sorted) * 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Baseline info
    ax.text(
        0.98,
        0.02,
        f"Baseline RMSE: {results['baseline_rmse']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="gray",
        style="italic",
    )

    plt.tight_layout()

    if save_path is None:
        save_path = str(RESULTS_DIR / "feature_importance.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature importance plot to {save_path}")
    return save_path


def log_explainability_to_mlflow(results: Dict, plot_path: str) -> None:
    """Log explainability artifacts to MLflow.

    Args:
        results: Permutation importance results.
        plot_path: Path to the feature importance plot.
    """
    try:
        import mlflow

        # Log importance values as metrics
        for name, importance in zip(results["feature_names"], results["importances_mean"]):
            mlflow.log_metric(f"feature_importance_{name.lower()}", importance)

        mlflow.log_metric("explainability_baseline_rmse", results["baseline_rmse"])

        # Log artifacts
        json_path = RESULTS_DIR / "permutation_importance.json"
        if Path(json_path).exists():
            mlflow.log_artifact(str(json_path), "explainability")
        if Path(plot_path).exists():
            mlflow.log_artifact(plot_path, "explainability")

        mlflow.set_tag("explainability_method", "permutation_importance")
        logger.info("Logged explainability artifacts to MLflow")

    except Exception as e:
        logger.warning(f"Could not log explainability to MLflow: {e}")
