"""LIME-based local explanations for the NVIDIA LSTM model.

Uses LIME (Local Interpretable Model-agnostic Explanations) to explain
individual predictions by approximating the LSTM locally with an
interpretable linear model.

Because LIME operates on flat tabular input while the LSTM requires 3-D
sequences (samples × timesteps × features), we implement an adapter that:
  1. Flattens the *last timestep* of each sequence into a 1-D feature vector.
  2. Perturbs that vector and reconstructs valid 3-D input before calling
     the LSTM.
  3. Returns per-feature contribution weights for the explained sample.

This provides *local* explanations (why did the model predict **this**
particular value?) which complement the *global* view given by permutation
importance.

References:
    - Ribeiro, Singh & Guestrin, "Why Should I Trust You?", KDD 2016
    - Molnar, "Interpretable Machine Learning", 2022, Ch. 9
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "explainability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _build_predict_fn(
    model: nn.Module,
    base_sequence: np.ndarray,
    device: torch.device,
    output_index: int = 0,
) -> callable:
    """Return a function ``f(X_flat) -> predictions`` usable by LIME.

    ``X_flat`` has shape ``(n_samples, n_features)`` where each row is
    a perturbation of the *last timestep* of ``base_sequence``.  The
    function splices each perturbation into a copy of the full sequence,
    runs the LSTM, and returns scalar outputs.

    Args:
        model: Trained PyTorch LSTM.
        base_sequence: Original 3-D sequence ``(1, seq_len, n_features)``.
        device: Torch device.
        output_index: Which output neuron to explain (default 0 = Close).

    Returns:
        Callable ``(np.ndarray) -> np.ndarray`` mapping
        ``(N, n_features)`` → ``(N,)``.
    """
    model.eval()
    seq = base_sequence.copy()  # (1, T, F)

    def _predict(X_flat: np.ndarray) -> np.ndarray:
        n = X_flat.shape[0]
        # Tile base sequence for every perturbed sample
        tiled = np.tile(seq, (n, 1, 1))  # (N, T, F)
        # Replace *last* timestep with LIME perturbation
        tiled[:, -1, :] = X_flat
        tensor = torch.FloatTensor(tiled).to(device)
        with torch.no_grad():
            preds = model(tensor).cpu().numpy()  # (N, output_size)
        # Return the column that corresponds to the explained output
        if preds.ndim == 1:
            return preds
        return preds[:, output_index]

    return _predict


# ────────────────────────────────────────────────────────────────────────
# Core LIME explanation
# ────────────────────────────────────────────────────────────────────────


def explain_with_lime(
    model: nn.Module,
    X: np.ndarray,
    sample_index: int = 0,
    feature_names: Optional[List[str]] = None,
    output_index: int = 0,
    num_features: Optional[int] = None,
    num_samples: int = 500,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict:
    """Explain a single LSTM prediction with LIME.

    Args:
        model: Trained PyTorch LSTM.
        X: Full dataset ``(n_samples, seq_len, n_features)``.
        sample_index: Which sample to explain.
        feature_names: Human-readable feature names.
        output_index: Which output neuron to explain (default 0).
        num_features: Max features in the explanation (default = all).
        num_samples: Number of LIME perturbation samples.
        device: Torch device.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with explanation details.
    """
    from lime.lime_tabular import LimeTabularExplainer

    if device is None:
        device = torch.device("cpu")

    n_features = X.shape[2]
    if feature_names is None:
        if n_features <= len(DEFAULT_FEATURE_NAMES):
            feature_names = DEFAULT_FEATURE_NAMES[:n_features]
        else:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

    if num_features is None:
        num_features = n_features

    # Flatten the last timestep of each sample as "training data" for LIME
    X_flat = X[:, -1, :]  # (n_samples, n_features)

    explainer = LimeTabularExplainer(
        training_data=X_flat,
        feature_names=feature_names,
        mode="regression",
        random_state=random_state,
    )

    sample_seq = X[sample_index : sample_index + 1]  # (1, T, F)
    sample_flat = X_flat[sample_index]  # (F,)

    predict_fn = _build_predict_fn(model, sample_seq, device, output_index)

    explanation = explainer.explain_instance(
        data_row=sample_flat,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    # Extract per-feature weights from the local model
    feature_weights: Dict[str, float] = {}
    for feat_name, weight in explanation.as_list():
        # LIME may modify feature names for binning; map back
        matched = False
        for fn in feature_names:
            if fn.lower() in feat_name.lower():
                feature_weights[fn] = float(weight)
                matched = True
                break
        if not matched:
            feature_weights[feat_name] = float(weight)

    # Get the model prediction for this sample
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(sample_seq).to(device)).cpu().numpy()
    predicted_value = float(pred[0, output_index]) if pred.ndim > 1 else float(pred[0])

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "lime",
        "sample_index": sample_index,
        "output_index": output_index,
        "predicted_value": predicted_value,
        "intercept": float(explanation.intercept[0]) if hasattr(explanation.intercept, '__iter__') else float(explanation.intercept),
        "local_r2": float(explanation.score),
        "num_samples": num_samples,
        "num_features": num_features,
        "feature_names": feature_names,
        "feature_weights": feature_weights,
        "feature_weights_sorted": dict(
            sorted(feature_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ),
    }

    # Save JSON
    output_path = RESULTS_DIR / "lime_explanation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved LIME explanation to {output_path}")

    return results


# ────────────────────────────────────────────────────────────────────────
# Batch explanations (aggregate local → global view)
# ────────────────────────────────────────────────────────────────────────


def explain_batch_with_lime(
    model: nn.Module,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_index: int = 0,
    n_explain: int = 20,
    num_samples: int = 300,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict:
    """Explain multiple samples and aggregate LIME weights.

    Averaging local weights across many samples yields a *global*
    importance ranking comparable to permutation importance, but
    richer because we also get per-sample local explanations.

    Args:
        model: Trained PyTorch LSTM.
        X: Dataset ``(n_samples, seq_len, n_features)``.
        y: Optional targets (for reference only).
        feature_names: Feature names.
        output_index: Output neuron to explain.
        n_explain: How many samples to explain.
        num_samples: LIME perturbation budget per sample.
        device: Torch device.
        random_state: Seed.

    Returns:
        Aggregated results with per-sample and mean weights.
    """
    if device is None:
        device = torch.device("cpu")

    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=min(n_explain, len(X)), replace=False)
    indices.sort()

    all_weights: Dict[str, List[float]] = {}
    per_sample: List[Dict] = []

    for idx in indices:
        expl = explain_with_lime(
            model=model,
            X=X,
            sample_index=int(idx),
            feature_names=feature_names,
            output_index=output_index,
            num_samples=num_samples,
            device=device,
            random_state=random_state + int(idx),
        )
        for fname, w in expl["feature_weights"].items():
            all_weights.setdefault(fname, []).append(abs(w))
        per_sample.append({
            "sample_index": int(idx),
            "predicted_value": expl["predicted_value"],
            "local_r2": expl["local_r2"],
            "feature_weights": expl["feature_weights"],
        })

    # Aggregate
    mean_abs_weights = {fn: float(np.mean(ws)) for fn, ws in all_weights.items()}
    std_abs_weights = {fn: float(np.std(ws)) for fn, ws in all_weights.items()}

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "lime_batch",
        "n_explained": len(indices),
        "num_samples_per_explanation": num_samples,
        "output_index": output_index,
        "feature_names": list(mean_abs_weights.keys()),
        "mean_abs_weights": mean_abs_weights,
        "std_abs_weights": std_abs_weights,
        "global_ranking": list(
            dict(sorted(mean_abs_weights.items(), key=lambda kv: kv[1], reverse=True)).keys()
        ),
        "per_sample_explanations": per_sample,
    }

    output_path = RESULTS_DIR / "lime_batch_explanation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved LIME batch explanation to {output_path}")

    return results


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────


def plot_lime_explanation(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> str:
    """Bar chart of LIME feature weights for a single sample.

    Args:
        results: Output from ``explain_with_lime``.
        save_path: Where to save the plot.
        figsize: Figure dimensions.

    Returns:
        Path to saved image.
    """
    weights = results["feature_weights"]
    names = list(weights.keys())
    values = [weights[n] for n in names]

    sorted_idx = np.argsort(np.abs(values))
    names_sorted = [names[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#EF4444" if v < 0 else "#22C55E" for v in values_sorted]
    ax.barh(range(len(names_sorted)), values_sorted, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=11)
    ax.set_xlabel("LIME Weight (local contribution)", fontsize=12)
    ax.set_title(
        f"LIME Explanation — Sample #{results.get('sample_index', '?')}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    ax.text(
        0.98, 0.02,
        f"Local R² = {results.get('local_r2', 0):.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="gray", style="italic",
    )

    plt.tight_layout()

    if save_path is None:
        save_path = str(RESULTS_DIR / "lime_explanation.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved LIME explanation plot to {save_path}")
    return save_path


def plot_lime_global(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> str:
    """Bar chart of mean |LIME weights| across multiple samples.

    Args:
        results: Output from ``explain_batch_with_lime``.
        save_path: Where to save.
        figsize: Figure dimensions.

    Returns:
        Path to saved image.
    """
    means = results["mean_abs_weights"]
    stds = results.get("std_abs_weights", {})
    names = list(means.keys())
    values = [means[n] for n in names]
    errs = [stds.get(n, 0) for n in names]

    sorted_idx = np.argsort(values)
    names_sorted = [names[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]
    errs_sorted = [errs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Purples(np.linspace(0.3, 0.8, len(names)))
    ax.barh(
        range(len(names_sorted)), values_sorted,
        xerr=errs_sorted, color=[colors[i] for i in range(len(names))],
        edgecolor="white", linewidth=0.5, capsize=4,
    )

    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=11)
    ax.set_xlabel("Mean |LIME Weight| (global importance)", fontsize=12)
    ax.set_title(
        f"LIME Global Feature Importance ({results.get('n_explained', '?')} samples)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = str(RESULTS_DIR / "lime_global_importance.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved LIME global importance plot to {save_path}")
    return save_path


def log_lime_to_mlflow(results: Dict, plot_path: str) -> None:
    """Log LIME artifacts to MLflow.

    Args:
        results: LIME results (single or batch).
        plot_path: Path to the plot image.
    """
    try:
        import mlflow

        method = results.get("method", "lime")

        if "mean_abs_weights" in results:
            # Batch / global
            for name, weight in results["mean_abs_weights"].items():
                mlflow.log_metric(f"lime_mean_abs_{name.lower()}", weight)
            mlflow.log_metric("lime_n_explained", results.get("n_explained", 0))
        else:
            # Single sample
            for name, weight in results.get("feature_weights", {}).items():
                mlflow.log_metric(f"lime_weight_{name.lower()}", weight)
            mlflow.log_metric("lime_local_r2", results.get("local_r2", 0))

        json_path = RESULTS_DIR / f"{method.replace(' ', '_')}_explanation.json"
        if json_path.exists():
            mlflow.log_artifact(str(json_path), "explainability")
        if Path(plot_path).exists():
            mlflow.log_artifact(plot_path, "explainability")

        mlflow.set_tag("explainability_lime", "true")
        logger.info("Logged LIME artifacts to MLflow")

    except Exception as e:
        logger.warning(f"Could not log LIME to MLflow: {e}")
