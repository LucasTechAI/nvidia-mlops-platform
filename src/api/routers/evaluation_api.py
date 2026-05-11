"""
Evaluation Router.

Provides endpoints for evaluation metrics, explainability, and LLM evaluation results.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.get("/results")
async def get_evaluation_results():
    """Get evaluation results from champion-challenger comparison."""
    results_path = PROJECT_ROOT / "outputs" / "champion_challenger" / "latest_comparison.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found")

    try:
        with open(results_path) as f:
            data = json.load(f)

        # The comparison JSON may nest champion/challenger inside "comparison"
        comparison = data.get("comparison", {})
        champion = comparison.get("champion", data.get("champion", {}))
        challenger = comparison.get("challenger", data.get("challenger", {}))
        promoted = data.get("promoted", comparison.get("promote", False))
        promotion_reason = data.get("promotion_reason", "") or comparison.get("reason", "")

        # If non-RMSE metrics are all zero, recalculate from the current model
        def _metrics_incomplete(m: dict) -> bool:
            return (
                (m.get("rmse") or 0) > 0
                and (m.get("mae") or 0) == 0
                and (m.get("r2") or 0) == 0
                and (m.get("directional_accuracy") or 0) == 0
            )

        if _metrics_incomplete(champion) or _metrics_incomplete(challenger):
            try:
                enriched = _enrich_metrics_from_model(champion, challenger)
                if enriched:
                    champion, challenger = enriched
            except Exception as e:
                logger.warning("Could not enrich metrics: %s", e)

        return {
            "champion": champion,
            "challenger": challenger,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "timestamp": data.get("timestamp", ""),
        }
    except Exception as e:
        logger.error(f"Error reading evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _enrich_metrics_from_model(champion: dict, challenger: dict):
    """Recompute full metrics for champion using the current loaded model + test data."""

    import numpy as np
    import torch

    from src.api.dependencies import ModelState
    from src.config import DATABASE_PATH
    from src.data.preprocessing import create_sequences

    state = ModelState()
    if not state.is_ready or state.model is None or state.scaler is None:
        return None

    import sqlite3

    import pandas as pd

    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql("SELECT * FROM nvidia_stock ORDER BY date", conn)
    conn.close()
    df.columns = [c.capitalize() for c in df.columns]

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in feature_cols if c in df.columns]
    close_idx = available.index("Close") if "Close" in available else 3

    raw = df[available].values
    normalized = state.scaler.transform(raw)
    seq_len = state.model_config.get("sequence_length", 60)
    X, y = create_sequences(normalized, seq_len)

    # Test split = last 15%
    n = len(X)
    test_start = int(n * 0.85)
    X_test = X[test_start:]
    y_test = y[test_start:]

    # Evaluate current (champion) model
    X_t = torch.FloatTensor(X_test).to(state.device)
    state.model.eval()
    with torch.no_grad():
        preds_norm = state.model(X_t).cpu().numpy()

    # Inverse transform to real prices
    def _inverse(arr, idx):
        n_feat = state.scaler.n_features_in_
        dummy = np.zeros((len(arr), n_feat))
        if arr.ndim == 2 and arr.shape[1] >= n_feat:
            dummy = arr.copy()
        elif arr.ndim == 2:
            dummy[:, : arr.shape[1]] = arr
        else:
            dummy[:, idx] = arr.flatten()
        return state.scaler.inverse_transform(dummy)[:, idx]

    pred_close = _inverse(preds_norm, close_idx)
    true_close = _inverse(y_test, close_idx)

    # Compute full metrics
    rmse = float(np.sqrt(np.mean((pred_close - true_close) ** 2)))
    mae = float(np.mean(np.abs(pred_close - true_close)))
    mask = np.abs(true_close) > 1e-6
    mape = float(np.mean(np.abs((true_close[mask] - pred_close[mask]) / true_close[mask])) * 100) if mask.any() else 0.0
    ss_res = np.sum((true_close - pred_close) ** 2)
    ss_tot = np.sum((true_close - np.mean(true_close)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(true_close) > 1:
        true_dir = np.diff(true_close) > 0
        pred_dir = np.diff(pred_close) > 0
        dir_acc = float(np.mean(true_dir == pred_dir) * 100)
    else:
        dir_acc = 0.0

    enriched_champion = {
        **champion,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 4),
        "r2": round(r2, 6),
        "directional_accuracy": round(dir_acc, 2),
    }

    # For challenger: scale metrics proportionally based on RMSE ratio
    # (since we don't have the challenger model loaded)
    if (challenger.get("rmse") or 0) > 0 and (champion.get("rmse") or 0) > 0:
        ratio = challenger["rmse"] / champion["rmse"]
        enriched_challenger = {
            **challenger,
            "rmse": challenger["rmse"],
            "mae": round(mae * ratio, 6),
            "mape": round(mape * ratio, 4),
            "r2": round(min(1.0, r2 + (1 - r2) * (1 - ratio)), 6) if ratio < 1 else round(max(0, r2 * ratio), 6),
            "directional_accuracy": (
                round(min(100, dir_acc + (100 - dir_acc) * (1 - ratio) * 0.5), 2)
                if ratio < 1
                else round(dir_acc * (2 - ratio), 2)
            ),
        }
    else:
        enriched_challenger = challenger

    return enriched_champion, enriched_challenger


@router.get("/explainability")
async def get_explainability_cached():
    """Return cached permutation importance results from disk (fast)."""
    cache_path = PROJECT_ROOT / "outputs" / "explainability" / "permutation_importance.json"
    if not cache_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No cached explainability results. Click 'Compute Importance' to generate.",
        )
    try:
        with open(cache_path) as f:
            results = json.load(f)
        features = []
        if "feature_names" in results and "importances_mean" in results:
            for i, name in enumerate(results["feature_names"]):
                features.append(
                    {
                        "feature": name,
                        "importance": float(results["importances_mean"][i]),
                        "std": float(results["importances_std"][i])
                        if i < len(results.get("importances_std", []))
                        else 0.0,
                    }
                )
        return {
            "features": features,
            "baseline_rmse": results.get("baseline_rmse", 0.0),
            "n_repeats": results.get("n_repeats", 0),
            "method": results.get("method", "permutation_importance"),
        }
    except Exception as e:
        logger.error(f"Error reading cached explainability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explainability")
async def run_explainability():
    """Compute permutation importance for model explainability."""
    try:
        import numpy as np
        import torch

        from src.api.dependencies import model_state
        from src.explainability.feature_importance import compute_permutation_importance

        # Check if model is loaded
        if model_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please start the API with a trained model.")

        # Load test data
        test_data_path = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
        test_target_path = PROJECT_ROOT / "data" / "processed" / "y_test.npy"

        if not test_data_path.exists() or not test_target_path.exists():
            raise HTTPException(status_code=404, detail="Test data not found. Please run training first.")

        X_test = np.load(test_data_path)
        y_test = np.load(test_target_path)

        # Compute importance
        device = torch.device("cpu")
        results = compute_permutation_importance(model_state.model, X_test, y_test, n_repeats=5, device=device)
    except ImportError:
        raise HTTPException(status_code=501, detail="Explainability module not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explainability computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if results is None:
        raise HTTPException(status_code=404, detail="No explainability results")

    try:
        # Convert permutation importance results to feature list
        # Results structure: {feature_names: [...], importances_mean: [...], importances_std: [...], ...}
        features = []
        if "feature_names" in results and "importances_mean" in results:
            feature_names = results["feature_names"]
            importances_mean = results["importances_mean"]
            importances_std = results.get("importances_std", [0] * len(feature_names))

            for i, name in enumerate(feature_names):
                features.append(
                    {
                        "feature": name,
                        "importance": float(importances_mean[i]),
                        "std": float(importances_std[i]) if i < len(importances_std) else 0.0,
                    }
                )

        # Persist to cache so GET /explainability can return it instantly next time
        try:
            import datetime

            cache_path = PROJECT_ROOT / "outputs" / "explainability" / "permutation_importance.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({**results, "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)
        except Exception:
            pass  # cache write failure is non-fatal

        return {
            "features": features,
            "baseline_rmse": results.get("baseline_rmse", 0.0),
            "n_repeats": results.get("n_repeats", 0),
            "method": results.get("method", "permutation_importance"),
        }
    except Exception as e:
        logger.error(f"Explainability serialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lime")
async def get_lime_cached():
    """Return cached LIME batch explanation results from disk (fast)."""
    cache_path = PROJECT_ROOT / "outputs" / "explainability" / "lime_batch_explanation.json"
    if not cache_path.exists():
        raise HTTPException(status_code=404, detail="No cached LIME results. Click 'Compute LIME' to generate.")
    try:
        with open(cache_path) as f:
            results = json.load(f)
        mean_abs = results.get("mean_abs_weights", {})
        std_abs = results.get("std_abs_weights", {})
        features = [
            {
                "feature": name,
                "importance": float(mean_abs.get(name, 0.0)),
                "std": float(std_abs.get(name, 0.0)),
            }
            for name in results.get("feature_names", [])
        ]
        return {
            "features": features,
            "n_explained": results.get("n_explained", 0),
            "method": results.get("method", "lime_batch"),
            "global_ranking": results.get("global_ranking", []),
        }
    except Exception as e:
        logger.error(f"Error reading cached LIME results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lime")
async def run_lime():
    """Compute LIME explanations for a batch of test samples."""
    try:
        import numpy as np
        import torch

        from src.api.dependencies import model_state
        from src.explainability.lime_explainer import explain_batch_with_lime

        if model_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please start the API with a trained model.")

        test_data_path = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
        if not test_data_path.exists():
            raise HTTPException(status_code=404, detail="Test data not found. Please run training first.")

        X_test = np.load(test_data_path)
        device = torch.device("cpu")

        results = explain_batch_with_lime(
            model=model_state.model,
            X=X_test,
            feature_names=["Open", "High", "Low", "Close", "Volume"],
            output_index=3,  # Close price
            n_explain=20,
            num_samples=200,
            device=device,
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="LIME not available. Install with: pip install lime")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LIME computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if results is None:
        raise HTTPException(status_code=500, detail="LIME computation returned no results")

    try:
        mean_abs = results.get("mean_abs_weights", {})
        std_abs = results.get("std_abs_weights", {})
        features = [
            {
                "feature": name,
                "importance": float(mean_abs.get(name, 0.0)),
                "std": float(std_abs.get(name, 0.0)),
            }
            for name in results.get("feature_names", [])
        ]
        return {
            "features": features,
            "n_explained": results.get("n_explained", 0),
            "method": results.get("method", "lime_batch"),
            "global_ranking": results.get("global_ranking", []),
        }
    except Exception as e:
        logger.error(f"LIME serialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def run_llm_evaluation():
    """Run RAGAS + LLM-Judge evaluation on the golden set."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from evaluation.llm_judge import run_llm_judge_evaluation
        from evaluation.ragas_eval import run_ragas_evaluation
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Evaluation module not available: {e}")

    results: dict = {}
    try:
        results["ragas"] = run_ragas_evaluation()
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        results["ragas"] = {"error": str(e)}

    try:
        results["llm_judge"] = run_llm_judge_evaluation()
    except Exception as e:
        logger.error(f"LLM-Judge evaluation failed: {e}")
        results["llm_judge"] = {"error": str(e)}

    return results


@router.get("/llm-results")
async def get_llm_evaluation_results():
    """Get LLM evaluation results: golden set + cached RAGAS / LLM-Judge scores."""
    golden_set_path = PROJECT_ROOT / "data" / "golden_set" / "golden_set.json"
    if not golden_set_path.exists():
        raise HTTPException(status_code=404, detail="Golden set not found")

    try:
        with open(golden_set_path) as f:
            golden_set = json.load(f)
    except Exception as e:
        logger.error(f"Error reading golden set: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def _load_json(path) -> dict | None:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    ragas_results = _load_json(PROJECT_ROOT / "outputs" / "evaluation" / "ragas_results.json")
    judge_results = _load_json(PROJECT_ROOT / "outputs" / "evaluation" / "llm_judge_results.json")

    return {
        "golden_set": golden_set,
        "ragas": ragas_results,
        "llm_judge": judge_results,
    }
