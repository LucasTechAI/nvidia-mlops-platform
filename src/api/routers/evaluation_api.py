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

        # Extract evaluation metrics
        champion = data.get("champion", {})
        challenger = data.get("challenger", {})

        return {
            "champion": champion,
            "challenger": challenger,
            "promoted": data.get("promoted", False),
            "promotion_reason": data.get("promotion_reason", ""),
            "timestamp": data.get("timestamp", ""),
        }
    except Exception as e:
        logger.error(f"Error reading evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explainability")
async def run_explainability():
    """Compute permutation importance for model explainability."""
    try:
        from src.explainability.feature_importance import compute_permutation_importance

        results = compute_permutation_importance()
    except ImportError:
        raise HTTPException(status_code=501, detail="Explainability module not available")
    except Exception as e:
        logger.error(f"Explainability computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if results is None:
        raise HTTPException(status_code=404, detail="No explainability results")

    try:
        # Serialize
        serializable = []
        if isinstance(results, list):
            for item in results:
                entry = {}
                for k, v in item.items():
                    if hasattr(v, "item"):
                        entry[k] = v.item()
                    else:
                        entry[k] = v
                serializable.append(entry)
        elif isinstance(results, dict):
            for k, v in results.items():
                if hasattr(v, "item"):
                    serializable.append({"feature": k, "importance": v.item()})
                elif isinstance(v, (int, float)):
                    serializable.append({"feature": k, "importance": v})
                else:
                    serializable.append({"feature": k, "importance": v})

        return {"features": serializable}
    except Exception as e:
        logger.error(f"Explainability serialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-results")
async def get_llm_evaluation_results():
    """Get LLM evaluation results from golden set."""
    # Check for golden set
    golden_set_path = PROJECT_ROOT / "data" / "golden_set" / "golden_set.json"
    if not golden_set_path.exists():
        raise HTTPException(status_code=404, detail="Golden set not found")

    try:
        with open(golden_set_path) as f:
            golden_set = json.load(f)
    except Exception as e:
        logger.error(f"Error reading golden set: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Check for existing evaluation results
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_results = []
    if eval_dir.exists():
        for eval_file in sorted(eval_dir.glob("*.json"), reverse=True):
            try:
                with open(eval_file) as f:
                    eval_results.append({"filename": eval_file.name, "data": json.load(f)})
            except Exception:
                continue

    return {
        "golden_set": golden_set,
        "evaluation_results": eval_results[:5],  # Last 5 evaluations
    }
