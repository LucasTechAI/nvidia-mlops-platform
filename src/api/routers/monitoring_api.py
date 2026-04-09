"""
Monitoring Router.

Provides endpoints for drift detection and champion-challenger comparison.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.post("/drift")
async def run_drift_detection():
    """Run drift detection using PSI and return results."""
    try:
        from src.monitoring.drift import detect_drift_from_db

        results = detect_drift_from_db()
        if results is None:
            raise HTTPException(status_code=404, detail="No data available for drift detection")

        # Serialize results
        serializable = {}
        for key, val in results.items():
            if hasattr(val, "tolist"):
                serializable[key] = val.tolist()
            elif hasattr(val, "item"):
                serializable[key] = val.item()
            elif isinstance(val, dict):
                inner = {}
                for k, v in val.items():
                    if hasattr(v, "item"):
                        inner[k] = v.item()
                    elif hasattr(v, "tolist"):
                        inner[k] = v.tolist()
                    else:
                        inner[k] = v
                serializable[key] = inner
            else:
                serializable[key] = val

        return serializable
    except ImportError:
        raise HTTPException(status_code=501, detail="Drift detection module not available")
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/champion-challenger")
async def get_champion_challenger():
    """Get latest champion-challenger comparison results."""
    results_path = PROJECT_ROOT / "outputs" / "champion_challenger" / "latest_comparison.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No champion-challenger results found")

    try:
        with open(results_path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading champion-challenger results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/champion-challenger/run")
async def run_champion_challenger():
    """Run champion-challenger pipeline."""
    try:
        from src.training.champion_challenger import run_champion_challenger_pipeline

        results = run_champion_challenger_pipeline()
        if results is None:
            raise HTTPException(status_code=500, detail="Pipeline returned no results")

        # Serialize
        serializable = {}
        for key, val in results.items():
            if hasattr(val, "item"):
                serializable[key] = val.item()
            elif hasattr(val, "tolist"):
                serializable[key] = val.tolist()
            else:
                serializable[key] = val

        return serializable
    except ImportError:
        raise HTTPException(status_code=501, detail="Champion-challenger module not available")
    except Exception as e:
        logger.error(f"Champion-challenger pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
