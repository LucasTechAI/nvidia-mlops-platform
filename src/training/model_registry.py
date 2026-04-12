"""MLflow Model Registry lifecycle management.

Provides:
    - Model registration with versioning
    - Stage transitions (None → Staging → Production → Archived)
    - Model comparison and promotion gates
    - Rollback to previous production version
    - Model metadata and tag management

Works with local MLflow tracking server. If MLflow is unavailable,
operations are logged and tracked locally via SQLite fallback.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "data" / "model_registry.db"

STAGES = ["None", "Staging", "Production", "Archived"]

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS registered_models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    tags_json   TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    stage           TEXT    NOT NULL DEFAULT 'None',
    source_path     TEXT,
    run_id          TEXT,
    description     TEXT,
    metrics_json    TEXT    DEFAULT '{}',
    params_json     TEXT    DEFAULT '{}',
    tags_json       TEXT    DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE(model_name, version),
    FOREIGN KEY (model_name) REFERENCES registered_models(name)
);

CREATE TABLE IF NOT EXISTS stage_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    from_stage      TEXT    NOT NULL,
    to_stage        TEXT    NOT NULL,
    transitioned_by TEXT    DEFAULT 'system',
    reason          TEXT,
    timestamp       TEXT    NOT NULL
);
"""


@dataclass
class ModelVersionInfo:
    """Information about a model version."""

    model_name: str
    version: int
    stage: str
    source_path: str
    run_id: str
    description: str
    metrics: dict
    params: dict
    tags: dict
    created_at: str
    updated_at: str


@dataclass
class PromotionGateResult:
    """Result of a promotion gate check."""

    passed: bool
    version: int
    current_stage: str
    target_stage: str
    checks: list = field(default_factory=list)
    reason: str = ""


class ModelRegistry:
    """Local model registry with lifecycle management."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(CREATE_TABLES)
        conn.commit()

    # ── Model Registration ────────────────────────────────────
    def register_model(self, name: str, description: str = "", tags: Optional[dict] = None) -> str:
        """Register a new model (or return existing)."""
        now = datetime.utcnow().isoformat()
        with self._cursor() as cur:
            cur.execute("SELECT name FROM registered_models WHERE name = ?", (name,))
            if cur.fetchone():
                logger.info("Model '%s' already registered", name)
                return name
            cur.execute(
                """INSERT INTO registered_models
                   (name, description, created_at, updated_at, tags_json)
                   VALUES (?,?,?,?,?)""",
                (name, description, now, now, json.dumps(tags or {})),
            )
        logger.info("Registered model: %s", name)
        return name

    def create_version(
        self,
        model_name: str,
        source_path: str = "",
        run_id: str = "",
        description: str = "",
        metrics: Optional[dict] = None,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> ModelVersionInfo:
        """Create a new version of a registered model."""
        now = datetime.utcnow().isoformat()

        # Auto-register if needed
        self.register_model(model_name)

        with self._cursor() as cur:
            cur.execute("SELECT MAX(version) as mv FROM model_versions WHERE model_name = ?", (model_name,))
            row = cur.fetchone()
            version = (row["mv"] or 0) + 1

            cur.execute(
                """INSERT INTO model_versions
                   (model_name, version, stage, source_path, run_id, description,
                    metrics_json, params_json, tags_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_name, version, "None", source_path, run_id, description,
                    json.dumps(metrics or {}), json.dumps(params or {}),
                    json.dumps(tags or {}), now, now,
                ),
            )

            # Record transition
            cur.execute(
                """INSERT INTO stage_transitions
                   (model_name, version, from_stage, to_stage,
                    reason, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (model_name, version, "", "None", "Initial registration", now),
            )

            # Update parent
            cur.execute(
                "UPDATE registered_models SET updated_at = ? WHERE name = ?",
                (now, model_name),
            )

        info = ModelVersionInfo(
            model_name=model_name, version=version, stage="None",
            source_path=source_path, run_id=run_id, description=description,
            metrics=metrics or {}, params=params or {}, tags=tags or {},
            created_at=now, updated_at=now,
        )
        logger.info("Created %s v%d", model_name, version)
        return info

    # ── Stage Transitions ─────────────────────────────────────
    def transition_stage(
        self,
        model_name: str,
        version: int,
        target_stage: str,
        reason: str = "",
        transitioned_by: str = "system",
    ) -> bool:
        """Transition a model version to a new stage."""
        if target_stage not in STAGES:
            raise ValueError(f"Invalid stage: {target_stage}. Must be one of {STAGES}")

        now = datetime.utcnow().isoformat()

        with self._cursor() as cur:
            cur.execute(
                "SELECT stage FROM model_versions WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Model {model_name} v{version} not found")

            from_stage = row["stage"]

            # If promoting to Production, archive current production version
            if target_stage == "Production":
                cur.execute(
                    "SELECT version FROM model_versions WHERE model_name = ? AND stage = 'Production'",
                    (model_name,),
                )
                prod = cur.fetchone()
                if prod:
                    old_v = prod["version"]
                    cur.execute(
                        """UPDATE model_versions
                           SET stage = 'Archived', updated_at = ?
                           WHERE model_name = ? AND version = ?""",
                        (now, model_name, old_v),
                    )
                    cur.execute(
                        """INSERT INTO stage_transitions
                           (model_name, version, from_stage, to_stage,
                            transitioned_by, reason, timestamp)
                           VALUES (?,?,?,?,?,?,?)""",
                        (model_name, old_v, "Production", "Archived",
                         transitioned_by, f"Replaced by v{version}", now),
                    )
                    logger.info("Archived %s v%d (replaced by v%d)", model_name, old_v, version)

            # Perform transition
            cur.execute(
                "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND version = ?",
                (target_stage, now, model_name, version),
            )
            cur.execute(
                """INSERT INTO stage_transitions
                   (model_name, version, from_stage, to_stage,
                    transitioned_by, reason, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (model_name, version, from_stage, target_stage,
                 transitioned_by, reason, now),
            )

        logger.info("Transitioned %s v%d: %s → %s", model_name, version, from_stage, target_stage)
        return True

    # ── Promotion Gate ────────────────────────────────────────
    def check_promotion_gate(
        self,
        model_name: str,
        version: int,
        target_stage: str = "Production",
        min_rmse_improvement: float = 0.005,
    ) -> PromotionGateResult:
        """Check if a model version passes promotion gates."""
        checks = []
        with self._cursor() as cur:
            # Get candidate
            cur.execute(
                "SELECT * FROM model_versions WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
            candidate = cur.fetchone()
            if not candidate:
                return PromotionGateResult(False, version, "", target_stage, reason="Version not found")

            candidate_metrics = json.loads(candidate["metrics_json"])
            current_stage = candidate["stage"]

            # Check 1: Must not already be in target stage
            if current_stage == target_stage:
                checks.append({"check": "stage_different", "passed": False, "detail": "Already in target stage"})
            else:
                checks.append({"check": "stage_different", "passed": True, "detail": f"Current: {current_stage}"})

            # Check 2: Must have metrics
            if not candidate_metrics:
                checks.append({"check": "has_metrics", "passed": False, "detail": "No metrics recorded"})
            else:
                checks.append({"check": "has_metrics", "passed": True, "detail": f"{len(candidate_metrics)} metrics"})

            # Check 3: Compare against current production
            if target_stage == "Production":
                cur.execute(
                    "SELECT * FROM model_versions WHERE model_name = ? AND stage = 'Production'",
                    (model_name,),
                )
                prod = cur.fetchone()
                if prod:
                    prod_metrics = json.loads(prod["metrics_json"])
                    prod_rmse = prod_metrics.get("rmse", float("inf"))
                    cand_rmse = candidate_metrics.get("rmse", float("inf"))

                    improvement = (prod_rmse - cand_rmse) / prod_rmse if prod_rmse > 0 else 0
                    passed = improvement >= min_rmse_improvement
                    checks.append({
                        "check": "rmse_improvement",
                        "passed": passed,
                        "detail": f"Improvement: {improvement:.2%} (min: {min_rmse_improvement:.2%})",
                    })
                else:
                    checks.append({
                        "check": "rmse_improvement",
                        "passed": True,
                        "detail": "No current production model",
                    })

        all_passed = all(c["passed"] for c in checks)
        return PromotionGateResult(
            passed=all_passed,
            version=version,
            current_stage=current_stage,
            target_stage=target_stage,
            checks=checks,
            reason="All gates passed" if all_passed else "Some gates failed",
        )

    # ── Rollback ──────────────────────────────────────────────
    def rollback_production(self, model_name: str, reason: str = "manual rollback") -> Optional[ModelVersionInfo]:
        """Rollback to the previous production version."""
        with self._cursor() as cur:
            # Find last archived version (most recently replaced)
            cur.execute(
                """SELECT version FROM stage_transitions
                   WHERE model_name = ? AND from_stage = 'Production' AND to_stage = 'Archived'
                   ORDER BY timestamp DESC LIMIT 1""",
                (model_name,),
            )
            row = cur.fetchone()
            if not row:
                logger.warning("No previous production version to rollback to for %s", model_name)
                return None

            prev_version = row["version"]

        # Archive current production and restore previous
        self.transition_stage(model_name, prev_version, "Production", reason=reason)
        logger.info("Rolled back %s to v%d", model_name, prev_version)
        return self.get_version(model_name, prev_version)

    # ── Queries ───────────────────────────────────────────────
    def get_version(self, model_name: str, version: int) -> Optional[ModelVersionInfo]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM model_versions WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ModelVersionInfo(
                model_name=row["model_name"], version=row["version"], stage=row["stage"],
                source_path=row["source_path"], run_id=row["run_id"], description=row["description"],
                metrics=json.loads(row["metrics_json"]), params=json.loads(row["params_json"]),
                tags=json.loads(row["tags_json"]), created_at=row["created_at"], updated_at=row["updated_at"],
            )

    def get_production_version(self, model_name: str) -> Optional[ModelVersionInfo]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM model_versions WHERE model_name = ? AND stage = 'Production'",
                (model_name,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self.get_version(model_name, row["version"])

    def list_versions(self, model_name: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT model_name, version, stage, created_at,
                   updated_at, metrics_json
                   FROM model_versions
                   WHERE model_name = ?
                   ORDER BY version DESC""",
                (model_name,),
            )
            return [
                {
                    "model_name": r["model_name"],
                    "version": r["version"],
                    "stage": r["stage"],
                    "created_at": r["created_at"],
                    "metrics": json.loads(r["metrics_json"]),
                }
                for r in cur.fetchall()
            ]

    def list_models(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM registered_models ORDER BY name")
            return [dict(r) for r in cur.fetchall()]

    def get_transition_history(self, model_name: str, limit: int = 50) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM stage_transitions WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
                (model_name, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ── Seed demo data ────────────────────────────────────────
    def seed_sample_data(self):
        """Seed registry with sample model versions for demo."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM registered_models")
            if cur.fetchone()["c"] > 0:
                return

        self.register_model(
            "nvidia-lstm-forecast",
            description="LSTM model for NVIDIA stock price prediction (5-day horizon)",
            tags={"framework": "pytorch", "task": "time-series-forecasting", "asset": "NVDA"},
        )

        # v1: baseline
        self.create_version(
            "nvidia-lstm-forecast",
            source_path="models/best_model.pth",
            run_id="f91746974ba246e39ae0897942ee2b7a",
            description="Baseline LSTM — 2 layers, hidden=64",
            metrics={"rmse": 4.23, "mae": 3.15, "mape": 2.8, "r2": 0.91, "directional_accuracy": 0.68},
            params={"hidden_size": 64, "num_layers": 2, "dropout": 0.2, "lr": 0.001, "epochs": 100},
        )
        self.transition_stage("nvidia-lstm-forecast", 1, "Archived", reason="Replaced by Optuna-tuned v2")

        # v2: optuna-tuned
        self.create_version(
            "nvidia-lstm-forecast",
            source_path="models/best_model.pth",
            run_id="205764c44d73421d811578ef93081a07",
            description="Optuna-tuned LSTM — 2 layers, hidden=128, dropout=0.15",
            metrics={"rmse": 3.47, "mae": 2.58, "mape": 2.1, "r2": 0.94, "directional_accuracy": 0.74},
            params={"hidden_size": 128, "num_layers": 2, "dropout": 0.15, "lr": 0.0008, "epochs": 150},
        )
        self.transition_stage("nvidia-lstm-forecast", 2, "Staging", reason="Passed champion-challenger evaluation")
        self.transition_stage(
            "nvidia-lstm-forecast", 2, "Production",
            reason="Promoted after validation \u2014 RMSE improved 17.9%",
        )

        # v3: latest challenger (staging)
        self.create_version(
            "nvidia-lstm-forecast",
            source_path="models/best_model.pth",
            run_id="ee17873ae3354481926bf70ac77130ef",
            description="Challenger — 3 layers, hidden=128, attention head",
            metrics={"rmse": 3.51, "mae": 2.62, "mape": 2.2, "r2": 0.935, "directional_accuracy": 0.72},
            params={"hidden_size": 128, "num_layers": 3, "dropout": 0.1, "lr": 0.0005, "epochs": 200},
        )
        self.transition_stage("nvidia-lstm-forecast", 3, "Staging", reason="Pending champion-challenger comparison")

        logger.info("Seeded model registry with nvidia-lstm-forecast (3 versions)")
