"""Explainability module for LSTM model interpretation."""

from src.explainability.feature_importance import (
    compute_permutation_importance,
    plot_feature_importance,
)
from src.explainability.lime_explainer import (
    explain_batch_with_lime,
    explain_with_lime,
    plot_lime_explanation,
    plot_lime_global,
)

__all__ = [
    "compute_permutation_importance",
    "plot_feature_importance",
    "explain_with_lime",
    "explain_batch_with_lime",
    "plot_lime_explanation",
    "plot_lime_global",
]
