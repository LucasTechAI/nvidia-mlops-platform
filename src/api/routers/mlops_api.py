"""API Router — Business Metrics, SLA, Feature Store, Model Registry, Canary Deploy, Cost Analysis.

Exposes the new monitoring and MLOps modules via REST endpoints.
"""

import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mlops"])

# ═══════════════════════════════════════════════════════════════
#  Business Metrics
# ═══════════════════════════════════════════════════════════════


@router.get("/business-metrics/snapshot")
async def business_snapshot():
    """Get current business metrics snapshot (P&L, ROI, Sharpe, etc.)."""
    try:
        from src.monitoring.business_metrics import BusinessMetricsTracker

        tracker = BusinessMetricsTracker.get_instance()
        snap = tracker.compute_snapshot()
        return asdict(snap)
    except Exception as e:
        logger.error("Business snapshot failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/business-metrics/pnl-history")
async def pnl_history(days: int = Query(30, ge=1, le=365)):
    """Get P&L time-series for charting."""
    try:
        from src.monitoring.business_metrics import BusinessMetricsTracker

        tracker = BusinessMetricsTracker.get_instance()
        return {"history": tracker.get_pnl_history(limit=days)}
    except Exception as e:
        logger.error("P&L history failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/business-metrics/daily-summaries")
async def daily_summaries(days: int = Query(30, ge=1, le=365)):
    """Get daily aggregated business summaries."""
    try:
        from src.monitoring.business_metrics import BusinessMetricsTracker

        tracker = BusinessMetricsTracker.get_instance()
        return {"summaries": tracker.get_daily_summaries(days=days)}
    except Exception as e:
        logger.error("Daily summaries failed: %s", e)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════
#  SLA Monitor
# ═══════════════════════════════════════════════════════════════


@router.get("/sla/report")
async def sla_report(period_minutes: int = Query(60, ge=1, le=10080)):
    """Get SLA compliance report for the given time window."""
    try:
        from src.monitoring.sla_monitor import SLAMonitor

        monitor = SLAMonitor.get_instance()
        report = monitor.compute_sla(period_minutes=period_minutes)
        return asdict(report)
    except Exception as e:
        logger.error("SLA report failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/sla/uptime-history")
async def sla_uptime_history(days: int = Query(7, ge=1, le=90)):
    """Get daily uptime history."""
    try:
        from src.monitoring.sla_monitor import SLAMonitor

        monitor = SLAMonitor.get_instance()
        return {"history": monitor.get_uptime_history(days=days)}
    except Exception as e:
        logger.error("Uptime history failed: %s", e)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════
#  Feature Store
# ═══════════════════════════════════════════════════════════════


@router.get("/feature-store/list")
async def list_feature_sets():
    """List all registered feature sets."""
    try:
        from src.data.feature_store import FeatureStore

        store = FeatureStore.get_instance()
        return {"feature_sets": store.list_feature_sets()}
    except Exception as e:
        logger.error("Feature store list failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/feature-store/{name}")
async def get_feature_set_info(name: str, version: Optional[int] = None):
    """Get metadata for a specific feature set."""
    try:
        from src.data.feature_store import FeatureStore

        store = FeatureStore.get_instance()
        meta = store.get_feature_set_meta(name, version)
        if not meta:
            raise HTTPException(404, f"Feature set '{name}' not found")
        return asdict(meta)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Feature store get failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/feature-store/{name}/lineage")
async def get_feature_lineage(name: str, version: Optional[int] = None):
    """Get lineage information for a feature set."""
    try:
        from src.data.feature_store import FeatureStore

        store = FeatureStore.get_instance()
        lineage = store.get_lineage(name, version)
        return {"lineage": lineage}
    except Exception as e:
        logger.error("Feature lineage failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/feature-store/{name}/preview")
async def preview_feature_set(name: str, version: Optional[int] = None, rows: int = Query(10, ge=1, le=100)):
    """Preview first N rows of a feature set."""
    try:
        from src.data.feature_store import FeatureStore

        store = FeatureStore.get_instance()
        df = store.get_feature_set(name, version)
        preview = df.head(rows).reset_index()
        return {
            "columns": list(preview.columns),
            "data": preview.to_dict(orient="records"),
            "total_rows": len(df),
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("Feature preview failed: %s", e)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════


@router.get("/model-registry/models")
async def list_registered_models():
    """List all registered models."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        return {"models": registry.list_models()}
    except Exception as e:
        logger.error("Model registry list failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/model-registry/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List all versions of a registered model."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        versions = registry.list_versions(model_name)
        return {"model_name": model_name, "versions": versions}
    except Exception as e:
        logger.error("Model versions list failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/model-registry/{model_name}/production")
async def get_production_model(model_name: str):
    """Get the current production model version."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        prod = registry.get_production_version(model_name)
        if not prod:
            raise HTTPException(404, f"No production version for '{model_name}'")
        return asdict(prod)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Production model get failed: %s", e)
        raise HTTPException(500, str(e))


@router.post("/model-registry/{model_name}/promote/{version}")
async def promote_model(model_name: str, version: int, target_stage: str = Query("Production")):
    """Promote a model version to a target stage (with gate checks)."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()

        # Check promotion gates
        gate = registry.check_promotion_gate(model_name, version, target_stage)
        if not gate.passed:
            return {"promoted": False, "gate_result": asdict(gate)}

        # Perform transition
        registry.transition_stage(model_name, version, target_stage, reason="API promotion")
        return {"promoted": True, "gate_result": asdict(gate)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Model promotion failed: %s", e)
        raise HTTPException(500, str(e))


@router.post("/model-registry/{model_name}/rollback")
async def rollback_model(model_name: str, reason: str = Query("manual rollback")):
    """Rollback to the previous production version."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        result = registry.rollback_production(model_name, reason)
        if not result:
            raise HTTPException(404, "No previous version to rollback to")
        return {"rolled_back": True, "restored_version": asdict(result)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rollback failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/model-registry/{model_name}/history")
async def model_transition_history(model_name: str, limit: int = Query(50, ge=1, le=200)):
    """Get stage transition history for a model."""
    try:
        from src.training.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        return {"history": registry.get_transition_history(model_name, limit)}
    except Exception as e:
        logger.error("Transition history failed: %s", e)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════
#  Canary Deployments
# ═══════════════════════════════════════════════════════════════


@router.get("/canary/deployments")
async def list_canary_deployments(model_name: Optional[str] = None, limit: int = Query(20, ge=1, le=100)):
    """List canary deployments."""
    try:
        from src.monitoring.canary_deploy import CanaryDeployManager

        mgr = CanaryDeployManager.get_instance()
        return {"deployments": mgr.list_deployments(model_name, limit)}
    except Exception as e:
        logger.error("Canary list failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/canary/{deployment_id}/status")
async def canary_status(deployment_id: str):
    """Get status of a canary deployment."""
    try:
        from src.monitoring.canary_deploy import CanaryDeployManager

        mgr = CanaryDeployManager.get_instance()
        status = mgr.get_status(deployment_id)
        if not status:
            raise HTTPException(404, f"Deployment '{deployment_id}' not found")
        return asdict(status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Canary status failed: %s", e)
        raise HTTPException(500, str(e))


@router.post("/canary/start")
async def start_canary(
    model_name: str = Query(...),
    canary_version: int = Query(...),
    baseline_version: int = Query(...),
):
    """Start a new canary deployment."""
    try:
        from src.monitoring.canary_deploy import CanaryDeployManager

        mgr = CanaryDeployManager.get_instance()
        dep_id = mgr.start_canary(model_name, canary_version, baseline_version)
        return {"deployment_id": dep_id, "state": "canary", "canary_weight": 5}
    except Exception as e:
        logger.error("Canary start failed: %s", e)
        raise HTTPException(500, str(e))


@router.post("/canary/{deployment_id}/evaluate")
async def evaluate_canary(deployment_id: str):
    """Evaluate current canary step and decide next action."""
    try:
        from src.monitoring.canary_deploy import CanaryDeployManager

        mgr = CanaryDeployManager.get_instance()
        action = mgr.evaluate_step(deployment_id)
        status = mgr.get_status(deployment_id)
        return {"action": action, "status": asdict(status) if status else None}
    except Exception as e:
        logger.error("Canary evaluate failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/canary/rollback-history")
async def canary_rollback_history(limit: int = Query(20, ge=1, le=100)):
    """Get rollback history across all canary deployments."""
    try:
        from src.monitoring.canary_deploy import CanaryDeployManager

        mgr = CanaryDeployManager.get_instance()
        return {"rollbacks": mgr.get_rollback_history(limit)}
    except Exception as e:
        logger.error("Rollback history failed: %s", e)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════
#  Cost Analysis
# ═══════════════════════════════════════════════════════════════

# ── Pricing constants (estimated, USD) ─────────────────────────
# Infrastructure — based on AWS/GCP equivalent on-demand pricing
INFRA_COSTS = {
    "gpu_training": {"name": "GPU Training (per run)", "unit_cost": 2.50, "unit": "run"},
    "api_server": {"name": "FastAPI Server (CPU)", "unit_cost": 0.048, "unit": "hour"},
    "mlflow_server": {"name": "MLflow Tracking", "unit_cost": 0.024, "unit": "hour"},
    "prometheus": {"name": "Prometheus + Grafana", "unit_cost": 0.035, "unit": "hour"},
    "storage_models": {"name": "Model Storage (S3/Disk)", "unit_cost": 0.023, "unit": "GB/month"},
    "storage_data": {"name": "Data Storage (raw + processed)", "unit_cost": 0.023, "unit": "GB/month"},
    "next_js": {"name": "Next.js Dashboard", "unit_cost": 0.012, "unit": "hour"},
    "optuna_db": {"name": "Optuna Dashboard + SQLite", "unit_cost": 0.010, "unit": "hour"},
}

# LLM pricing (OpenRouter rates for common models, per 1M tokens)
LLM_PRICING = {
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40, "name": "Gemini 2.0 Flash"},
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00, "name": "Gemini 2.5 Pro"},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00, "name": "GPT-4o"},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "name": "GPT-4o Mini"},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00, "name": "Claude Sonnet 4"},
    "meta-llama/llama-4-maverick": {"input": 0.20, "output": 0.60, "name": "Llama 4 Maverick"},
}


@router.get("/cost-analysis")
async def cost_analysis(days: int = Query(30, ge=1, le=365)):
    """Estimate infrastructure and LLM costs for the platform."""
    try:
        # ── Determine active LLM model ────────────────
        llm_model = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
        llm_provider = os.getenv("LLM_PROVIDER", "openrouter")
        model_pricing = LLM_PRICING.get(llm_model, LLM_PRICING["google/gemini-2.0-flash-001"])

        # ── Real LLM token usage from MLflow traces ──────────
        cutoff_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        mlflow_db = Path(__file__).resolve().parent.parent.parent.parent / "mlruns" / "mlflow.db"

        ragas_tokens_input = 0
        ragas_tokens_output = 0
        judge_tokens_input = 0
        judge_tokens_output = 0
        agent_tokens_input = 0
        agent_tokens_output = 0
        daily_token_map: dict = {}
        _using_real_tokens = False

        try:
            import json
            import sqlite3 as _sqlite3

            con = _sqlite3.connect(str(mlflow_db))
            rows = con.execute(
                """
                SELECT trm.key, trm.value, ti.timestamp_ms
                FROM trace_request_metadata trm
                JOIN trace_info ti ON trm.request_id = ti.request_id
                WHERE trm.key IN (
                    'mlflow.trace.tokenUsage',
                    'mlflow.trace.inputs',
                    'mlflow.trace.outputs'
                )
                  AND ti.timestamp_ms >= ?
                """,
                (cutoff_ms,),
            ).fetchall()
            con.close()

            for key, value, ts_ms in rows:
                day_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
                if key == "mlflow.trace.tokenUsage":
                    try:
                        usage = json.loads(value) if isinstance(value, str) else value
                        inp = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
                        out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
                        agent_tokens_input += inp
                        agent_tokens_output += out
                        if day_str not in daily_token_map:
                            daily_token_map[day_str] = {"input": 0, "output": 0}
                        daily_token_map[day_str]["input"] += inp
                        daily_token_map[day_str]["output"] += out
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass

            _using_real_tokens = True
        except Exception as _mlflow_exc:
            logger.warning("Could not read MLflow token usage, falling back to estimate: %s", _mlflow_exc)
            # Fallback estimates
            eval_runs_per_month = 10
            ragas_tokens_input = eval_runs_per_month * 4 * 25 * 300
            ragas_tokens_output = eval_runs_per_month * 4 * 25 * 100
            judge_tokens_input = eval_runs_per_month * 25 * 400
            judge_tokens_output = eval_runs_per_month * 25 * 150
            agent_tokens_input = 50 * days * 800
            agent_tokens_output = 50 * days * 300

        total_input_tokens = ragas_tokens_input + judge_tokens_input + agent_tokens_input
        total_output_tokens = ragas_tokens_output + judge_tokens_output + agent_tokens_output

        llm_input_cost = (total_input_tokens / 1_000_000) * model_pricing["input"]
        llm_output_cost = (total_output_tokens / 1_000_000) * model_pricing["output"]
        llm_total = llm_input_cost + llm_output_cost

        # ── Estimate infra costs ──────────────────────
        hours = days * 24

        # Training: estimate runs from model registry
        training_runs = 3  # default seed count
        try:
            from src.training.model_registry import ModelRegistry

            reg = ModelRegistry.get_instance()
            versions = reg.list_versions("nvidia-lstm-forecast")
            training_runs = max(len(versions), 3)
        except Exception:
            pass

        # SLA data for request count
        total_requests = 0
        try:
            from src.monitoring.sla_monitor import SLAMonitor

            sla = SLAMonitor.get_instance()
            report = sla.get_report(period_minutes=days * 1440)
            total_requests = (
                report.get("total_requests", 0) if isinstance(report, dict) else getattr(report, "total_requests", 0)
            )
        except Exception:
            total_requests = 5000 * days  # fallback estimate

        infra_breakdown = [
            {
                "name": "GPU Training",
                "quantity": training_runs,
                "unit": "runs",
                "unit_cost": INFRA_COSTS["gpu_training"]["unit_cost"],
                "total": round(training_runs * INFRA_COSTS["gpu_training"]["unit_cost"], 2),
            },
            {
                "name": "FastAPI Server",
                "quantity": hours,
                "unit": "hours",
                "unit_cost": INFRA_COSTS["api_server"]["unit_cost"],
                "total": round(hours * INFRA_COSTS["api_server"]["unit_cost"], 2),
            },
            {
                "name": "MLflow Tracking",
                "quantity": hours,
                "unit": "hours",
                "unit_cost": INFRA_COSTS["mlflow_server"]["unit_cost"],
                "total": round(hours * INFRA_COSTS["mlflow_server"]["unit_cost"], 2),
            },
            {
                "name": "Prometheus + Grafana",
                "quantity": hours,
                "unit": "hours",
                "unit_cost": INFRA_COSTS["prometheus"]["unit_cost"],
                "total": round(hours * INFRA_COSTS["prometheus"]["unit_cost"], 2),
            },
            {
                "name": "Next.js Dashboard",
                "quantity": hours,
                "unit": "hours",
                "unit_cost": INFRA_COSTS["next_js"]["unit_cost"],
                "total": round(hours * INFRA_COSTS["next_js"]["unit_cost"], 2),
            },
            {
                "name": "Optuna Dashboard",
                "quantity": hours,
                "unit": "hours",
                "unit_cost": INFRA_COSTS["optuna_db"]["unit_cost"],
                "total": round(hours * INFRA_COSTS["optuna_db"]["unit_cost"], 2),
            },
            {
                "name": "Model Storage",
                "quantity": 0.5,
                "unit": "GB",
                "unit_cost": INFRA_COSTS["storage_models"]["unit_cost"],
                "total": round(0.5 * INFRA_COSTS["storage_models"]["unit_cost"] * (days / 30), 2),
            },
            {
                "name": "Data Storage",
                "quantity": 0.2,
                "unit": "GB",
                "unit_cost": INFRA_COSTS["storage_data"]["unit_cost"],
                "total": round(0.2 * INFRA_COSTS["storage_data"]["unit_cost"] * (days / 30), 2),
            },
        ]

        infra_total = sum(item["total"] for item in infra_breakdown)

        llm_breakdown = [
            {
                "name": "RAGAS Evaluation (input)",
                "tokens": ragas_tokens_input,
                "cost": round((ragas_tokens_input / 1_000_000) * model_pricing["input"], 4),
            },
            {
                "name": "RAGAS Evaluation (output)",
                "tokens": ragas_tokens_output,
                "cost": round((ragas_tokens_output / 1_000_000) * model_pricing["output"], 4),
            },
            {
                "name": "LLM-Judge (input)",
                "tokens": judge_tokens_input,
                "cost": round((judge_tokens_input / 1_000_000) * model_pricing["input"], 4),
            },
            {
                "name": "LLM-Judge (output)",
                "tokens": judge_tokens_output,
                "cost": round((judge_tokens_output / 1_000_000) * model_pricing["output"], 4),
            },
            {
                "name": "RAG Agent (input)" + (" (real)" if _using_real_tokens else " (estimated)"),
                "tokens": agent_tokens_input,
                "cost": round((agent_tokens_input / 1_000_000) * model_pricing["input"], 4),
            },
            {
                "name": "RAG Agent (output)" + (" (real)" if _using_real_tokens else " (estimated)"),
                "tokens": agent_tokens_output,
                "cost": round((agent_tokens_output / 1_000_000) * model_pricing["output"], 4),
            },
        ]

        # ── Daily cost history (real per-day tokens where available) ──
        daily_cost_history = []
        day_infra = infra_total / days
        for i in range(min(days, 60)):
            day = datetime.utcnow() - timedelta(days=days - 1 - i)
            day_str = day.strftime("%Y-%m-%d")
            if day_str in daily_token_map:
                d_inp = daily_token_map[day_str]["input"]
                d_out = daily_token_map[day_str]["output"]
                day_llm = (d_inp / 1_000_000) * model_pricing["input"] + (d_out / 1_000_000) * model_pricing["output"]
            else:
                day_llm = llm_total / days
            daily_cost_history.append(
                {
                    "date": day_str,
                    "infra": round(day_infra, 2),
                    "llm": round(day_llm, 4),
                    "total": round(day_infra + day_llm, 2),
                }
            )

        # ── Model comparison (what if you used a different model?) ──
        model_comparison = []
        for model_id, pricing in LLM_PRICING.items():
            input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
            model_comparison.append(
                {
                    "model": pricing["name"],
                    "model_id": model_id,
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "total_cost": round(input_cost + output_cost, 4),
                    "is_current": model_id == llm_model,
                }
            )
        model_comparison.sort(key=lambda x: x["total_cost"])

        grand_total = round(infra_total + llm_total, 2)

        return {
            "period_days": days,
            "grand_total": grand_total,
            "infra_total": round(infra_total, 2),
            "llm_total": round(llm_total, 4),
            "infra_pct": round(infra_total / grand_total * 100, 1) if grand_total > 0 else 0,
            "llm_pct": round(llm_total / grand_total * 100, 1) if grand_total > 0 else 0,
            "current_model": model_pricing["name"],
            "current_model_id": llm_model,
            "provider": llm_provider,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_requests": total_requests,
            "training_runs": training_runs,
            "infra_breakdown": infra_breakdown,
            "llm_breakdown": llm_breakdown,
            "daily_history": daily_cost_history,
            "model_comparison": model_comparison,
        }
    except Exception as e:
        logger.error("Cost analysis failed: %s", e)
        raise HTTPException(500, str(e))
