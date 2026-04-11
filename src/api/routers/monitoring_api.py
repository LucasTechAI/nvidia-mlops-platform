"""
Monitoring Router.

Provides endpoints for drift detection and champion-challenger comparison.
"""

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _deep_serialize(obj: object) -> object:
    """Recursively convert numpy/non-JSON types into JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(v) for v in obj]
    # Fallback – let json handle it or convert to str
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _get_training_cutoff_date() -> str | None:
    """Approximate training-data cutoff from the model file’s modification time.

    The model checkpoint’s *mtime* marks when training finished.  Data
    in the database with dates **after** this is considered
    *post-training / production* data.
    """
    from datetime import datetime as _dt

    try:
        model_path = PROJECT_ROOT / "models" / "best_model.pth"
        if model_path.exists():
            mtime = _dt.fromtimestamp(model_path.stat().st_mtime)
            return mtime.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


@router.post("/drift")
async def run_drift_detection():
    """Run drift detection using PSI and return results."""
    try:
        from src.monitoring.drift import detect_drift_from_db

        training_date = _get_training_cutoff_date()
        results = detect_drift_from_db(training_cutoff_date=training_date)
    except ImportError:
        raise HTTPException(status_code=501, detail="Drift detection module not available")
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if results is None:
        raise HTTPException(status_code=404, detail="No data available for drift detection")

    try:
        return _deep_serialize(results)
    except Exception as e:
        logger.error(f"Drift serialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/all-triggers")
async def run_all_triggers():
    """Run all three retrain triggers and return a combined report.

    Triggers:
      1. Data Drift (PSI) – input distribution shift
      2. Model Staleness – model file age ≥ 30 days
      3. Prediction CI Breach – actuals outside 95% prediction CI

    For trigger 3, we attempt to run a quick backtest to obtain
    predictions vs actuals.  If the model is not available the
    trigger is reported as "skipped".
    """
    try:
        from src.monitoring.drift import detect_all_triggers
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Drift detection module not available",
        )

    # ── Determine training cutoff date ──
    training_date = _get_training_cutoff_date()

    # ── Try to obtain predictions / actuals for CI-breach trigger ──
    # The model forecasts up to 30 days ahead.  CI breach should test
    # ONLY those ~30 days right after training — checking whether actual
    # prices fell outside the model's 95% confidence interval.
    # This matches the same ±30-day window used by the PSI trigger.
    forecast_days = 30
    predictions = None
    actuals = None
    try:
        import sqlite3

        import pandas as pd
        import torch

        from src.api.dependencies import ModelState
        from src.config import DATABASE_PATH

        state = ModelState()
        if state.is_ready and state.model is not None and state.scaler is not None:
            state.model.eval()

            conn = sqlite3.connect(DATABASE_PATH)
            df = pd.read_sql("SELECT * FROM nvidia_stock ORDER BY date", conn)
            conn.close()
            df.columns = [c.capitalize() for c in df.columns]

            feature_columns = ["Open", "High", "Low", "Close", "Volume"]
            available = [c for c in feature_columns if c in df.columns]

            # Parse dates and determine the post-training window
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)

            seq_len = state.model_config.get("sequence_length", 60)
            n_features = state.scaler.n_features_in_
            close_idx = 3

            # Determine the backtest window: only the 30 days after training
            if training_date and "Date" in df.columns:
                cutoff = pd.Timestamp(training_date)
                cur_end = cutoff + pd.Timedelta(days=forecast_days)
                post_mask = (df["Date"] > cutoff) & (df["Date"] <= cur_end)
                n_post = int(post_mask.sum())

                if n_post >= 5:
                    # Backtest only on these post-training rows
                    post_indices = df.index[post_mask].tolist()
                    backtest_days = len(post_indices)
                else:
                    # Fallback: use the last `forecast_days` rows of the dataset
                    backtest_days = min(forecast_days, len(df) - seq_len)
                    post_indices = list(range(len(df) - backtest_days, len(df)))
            else:
                # No training date — use last `forecast_days` rows
                backtest_days = min(forecast_days, len(df) - seq_len)
                post_indices = list(range(len(df) - backtest_days, len(df)))

            feature_data = df[available].values
            normalized = state.scaler.transform(feature_data)

            if backtest_days >= 5:
                pred_list, act_list = [], []
                for row_idx in post_indices:
                    # We need `seq_len` rows BEFORE this row to form the input
                    seq_start = row_idx - seq_len
                    if seq_start < 0:
                        continue
                    seq = normalized[seq_start:row_idx]
                    if len(seq) < seq_len:
                        continue
                    seq_t = torch.FloatTensor(seq).unsqueeze(0).to(state.device)
                    with torch.no_grad():
                        p = state.model(seq_t)
                    p_np = p.cpu().numpy().flatten()
                    dummy = np.zeros((1, n_features))
                    dummy[0, : len(p_np)] = p_np
                    pred_price = float(state.scaler.inverse_transform(dummy)[0, close_idx])
                    actual_price = float(df["Close"].iloc[row_idx])
                    pred_list.append(pred_price)
                    act_list.append(actual_price)
                if len(pred_list) >= 5:
                    predictions = np.array(pred_list)
                    actuals = np.array(act_list)
                    logger.info(
                        "CI-breach: obtained %d prediction-actual pairs for %dd post-training window",
                        len(pred_list), forecast_days,
                    )
        else:
            logger.info("CI-breach: model not loaded, skipping predictions")
    except Exception as e:
        logger.warning("Could not obtain predictions for CI-breach: %s", e)

    # ── Run all triggers ──
    try:
        report = detect_all_triggers(
            predictions=predictions,
            actuals=actuals,
            training_cutoff_date=training_date,
            save_results=True,
        )
    except Exception as e:
        logger.error(f"All-triggers detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        return _deep_serialize(report)
    except Exception as e:
        logger.error(f"All-triggers serialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/history")
async def get_runs_history():
    """Return historical model runs from all MLflow experiments.

    Scans both file-based and SQLite-backed MLflow stores to collect
    every training run with its metrics & hyperparameters.
    """
    import os
    from datetime import datetime as _dt

    import yaml

    mlruns_root = PROJECT_ROOT / "mlruns"
    runs_list: list[dict] = []

    # ── 1. File-based experiments (folders with meta.yaml) ──
    if mlruns_root.exists():
        for exp_dir in sorted(mlruns_root.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_meta = exp_dir / "meta.yaml"
            exp_name = exp_dir.name
            if exp_meta.exists():
                try:
                    with open(exp_meta) as f:
                        em = yaml.safe_load(f)
                    exp_name = em.get("name", exp_dir.name)
                except Exception:
                    pass

            for run_dir in sorted(exp_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name in ("models",):
                    continue
                meta_path = run_dir / "meta.yaml"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path) as f:
                        meta = yaml.safe_load(f)
                except Exception:
                    continue

                run_id = run_dir.name
                start_ms = meta.get("start_time", 0)
                end_ms = meta.get("end_time", 0)

                # Read metrics
                metrics: dict[str, float] = {}
                metrics_dir = run_dir / "metrics"
                if metrics_dir.is_dir():
                    for mf in metrics_dir.iterdir():
                        try:
                            lines = mf.read_text().strip().splitlines()
                            if lines:
                                metrics[mf.name] = float(lines[-1].split()[1])
                        except Exception:
                            pass

                # Read params
                params: dict[str, str] = {}
                params_dir = run_dir / "params"
                if params_dir.is_dir():
                    for pf in params_dir.iterdir():
                        try:
                            params[pf.name] = pf.read_text().strip()
                        except Exception:
                            pass

                runs_list.append({
                    "run_id": run_id,
                    "run_name": meta.get("run_name", "—"),
                    "experiment": exp_name,
                    "status": "FINISHED" if meta.get("status") == 3 else str(meta.get("status", "?")),
                    "start_time": _dt.fromtimestamp(start_ms / 1000).isoformat() if start_ms else None,
                    "end_time": _dt.fromtimestamp(end_ms / 1000).isoformat() if end_ms else None,
                    "duration_s": round((end_ms - start_ms) / 1000, 1) if end_ms and start_ms else None,
                    "metrics": {k: round(v, 6) for k, v in metrics.items()},
                    "params": params,
                    "source": "file",
                })

    # ── 2. SQLite-backed runs ──
    mlflow_db = mlruns_root / "mlflow.db"
    if mlflow_db.exists():
        import sqlite3

        seen_ids = {r["run_id"] for r in runs_list}
        try:
            conn = sqlite3.connect(str(mlflow_db))
            cur = conn.cursor()

            # Map experiment ids to names
            cur.execute("SELECT experiment_id, name FROM experiments")
            exp_map = dict(cur.fetchall())

            cur.execute(
                "SELECT run_uuid, name, status, start_time, end_time, experiment_id "
                "FROM runs ORDER BY start_time DESC"
            )
            for row in cur.fetchall():
                rid, rname, status, start_ms, end_ms, eid = row
                if rid in seen_ids:
                    continue

                cur2 = conn.cursor()
                cur2.execute("SELECT key, value FROM latest_metrics WHERE run_uuid=?", (rid,))
                metrics = {k: round(v, 6) for k, v in cur2.fetchall()}

                cur2.execute("SELECT key, value FROM params WHERE run_uuid=?", (rid,))
                params = dict(cur2.fetchall())

                status_str = {
                    "FINISHED": "FINISHED", "RUNNING": "RUNNING",
                    "FAILED": "FAILED", "KILLED": "KILLED",
                }.get(status, status)

                runs_list.append({
                    "run_id": rid,
                    "run_name": rname or "—",
                    "experiment": exp_map.get(str(eid), str(eid)),
                    "status": status_str,
                    "start_time": _dt.fromtimestamp(start_ms / 1000).isoformat() if start_ms else None,
                    "end_time": _dt.fromtimestamp(end_ms / 1000).isoformat() if end_ms else None,
                    "duration_s": round((end_ms - start_ms) / 1000, 1) if end_ms and start_ms else None,
                    "metrics": metrics,
                    "params": params,
                    "source": "db",
                })
            conn.close()
        except Exception as e:
            logger.warning("Could not read mlflow.db: %s", e)

    # Sort by start_time descending
    runs_list.sort(key=lambda r: r.get("start_time") or "", reverse=True)

    return {"runs": runs_list, "total": len(runs_list)}


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
        from src.training.champion_challenger import run_champion_challenger

        results = run_champion_challenger()
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
