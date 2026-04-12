"""Canary Deployment & Automated Rollback engine.

Implements:
    - Gradual traffic shifting (canary weight ramp-up)
    - Health-check monitoring during rollout
    - Automated rollback on error/latency thresholds
    - Deployment state machine (pending → canary → promoted / rolled-back)

Designed to work with the SLA Monitor and Model Registry modules.
"""

import json
import logging
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "data" / "canary_deploy.db"


class DeploymentState(str, Enum):
    PENDING = "pending"
    CANARY = "canary"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# Default canary configuration
DEFAULT_CANARY_CONFIG = {
    "initial_weight_pct": 5,       # Start with 5% traffic
    "weight_step_pct": 10,         # Increase by 10% each step
    "step_interval_sec": 300,      # 5 min between steps
    "max_error_rate_pct": 2.0,     # Rollback if > 2% errors
    "max_latency_p95_ms": 600,     # Rollback if p95 > 600ms
    "min_requests_per_step": 10,   # Wait for at least 10 requests per step
    "promotion_threshold_pct": 80, # Promote when canary handles 80%+ traffic
}

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS deployments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id   TEXT    NOT NULL UNIQUE,
    model_name      TEXT    NOT NULL,
    canary_version  INTEGER NOT NULL,
    baseline_version INTEGER NOT NULL,
    state           TEXT    NOT NULL DEFAULT 'pending',
    canary_weight   REAL    NOT NULL DEFAULT 0,
    config_json     TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    completed_at    TEXT,
    result_json     TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS deployment_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id   TEXT    NOT NULL,
    step_number     INTEGER NOT NULL,
    canary_weight   REAL    NOT NULL,
    canary_requests INTEGER DEFAULT 0,
    canary_errors   INTEGER DEFAULT 0,
    canary_p95_ms   REAL    DEFAULT 0,
    baseline_requests INTEGER DEFAULT 0,
    baseline_errors INTEGER DEFAULT 0,
    baseline_p95_ms REAL    DEFAULT 0,
    health_ok       INTEGER DEFAULT 1,
    action          TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
);

CREATE TABLE IF NOT EXISTS rollback_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id   TEXT    NOT NULL,
    reason          TEXT    NOT NULL,
    canary_version  INTEGER NOT NULL,
    rolled_back_to  INTEGER NOT NULL,
    error_rate_pct  REAL,
    p95_latency_ms  REAL,
    timestamp       TEXT    NOT NULL
);
"""


@dataclass
class CanaryStatus:
    """Current status of a canary deployment."""

    deployment_id: str
    model_name: str
    canary_version: int
    baseline_version: int
    state: str
    canary_weight: float
    steps_completed: int
    total_canary_requests: int
    total_canary_errors: int
    current_error_rate: float
    current_p95_ms: float
    started_at: str
    elapsed_minutes: float


@dataclass
class DeploymentResult:
    """Final result of a deployment."""

    deployment_id: str
    state: str
    canary_version: int
    baseline_version: int
    promoted: bool
    total_steps: int
    total_requests: int
    final_error_rate: float
    final_p95_ms: float
    duration_minutes: float
    rollback_reason: str = ""


class CanaryDeployManager:
    """Manages canary deployments with automated rollback."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "CanaryDeployManager":
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

    # ── Start Deployment ──────────────────────────────────────
    def start_canary(
        self,
        model_name: str,
        canary_version: int,
        baseline_version: int,
        config: Optional[dict] = None,
    ) -> str:
        """Start a new canary deployment."""
        deploy_config = DEFAULT_CANARY_CONFIG.copy()
        if config:
            deploy_config.update(config)

        deployment_id = f"deploy-{model_name}-v{canary_version}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        now = datetime.utcnow().isoformat()
        initial_weight = deploy_config["initial_weight_pct"]

        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO deployments
                   (deployment_id, model_name, canary_version, baseline_version, state, canary_weight, config_json, started_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (deployment_id, model_name, canary_version, baseline_version, DeploymentState.CANARY.value,
                 initial_weight, json.dumps(deploy_config), now, now),
            )
            # Record first step
            cur.execute(
                """INSERT INTO deployment_steps
                   (deployment_id, step_number, canary_weight, action, timestamp)
                   VALUES (?,?,?,?,?)""",
                (deployment_id, 1, initial_weight, "start_canary", now),
            )

        logger.info("Started canary deploy %s: v%d (canary) vs v%d (baseline) @ %d%%",
                     deployment_id, canary_version, baseline_version, initial_weight)
        return deployment_id

    # ── Route request ─────────────────────────────────────────
    def route_request(self, deployment_id: str) -> str:
        """Decide whether to route a request to canary or baseline.

        Returns 'canary' or 'baseline'.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT state, canary_weight FROM deployments WHERE deployment_id = ?",
                (deployment_id,),
            )
            row = cur.fetchone()
            if not row or row["state"] != DeploymentState.CANARY.value:
                return "baseline"

            return "canary" if random.random() * 100 < row["canary_weight"] else "baseline"

    # ── Record request outcome ────────────────────────────────
    def record_request_outcome(
        self,
        deployment_id: str,
        target: str,  # 'canary' or 'baseline'
        latency_ms: float,
        is_error: bool = False,
    ):
        """Record the outcome of a routed request."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT step_number FROM deployment_steps WHERE deployment_id = ? ORDER BY step_number DESC LIMIT 1",
                (deployment_id,),
            )
            row = cur.fetchone()
            if not row:
                return

            step = row["step_number"]
            if target == "canary":
                cur.execute(
                    """UPDATE deployment_steps
                       SET canary_requests = canary_requests + 1,
                           canary_errors = canary_errors + ?,
                           canary_p95_ms = MAX(canary_p95_ms, ?)
                       WHERE deployment_id = ? AND step_number = ?""",
                    (1 if is_error else 0, latency_ms, deployment_id, step),
                )
            else:
                cur.execute(
                    """UPDATE deployment_steps
                       SET baseline_requests = baseline_requests + 1,
                           baseline_errors = baseline_errors + ?,
                           baseline_p95_ms = MAX(baseline_p95_ms, ?)
                       WHERE deployment_id = ? AND step_number = ?""",
                    (1 if is_error else 0, latency_ms, deployment_id, step),
                )

    # ── Evaluate step ─────────────────────────────────────────
    def evaluate_step(self, deployment_id: str) -> str:
        """Evaluate current step and decide: ramp_up, hold, promote, or rollback.

        Returns the action taken.
        """
        with self._cursor() as cur:
            cur.execute("SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,))
            deploy = cur.fetchone()
            if not deploy or deploy["state"] != DeploymentState.CANARY.value:
                return "no_action"

            config = json.loads(deploy["config_json"])

            cur.execute(
                "SELECT * FROM deployment_steps WHERE deployment_id = ? ORDER BY step_number DESC LIMIT 1",
                (deployment_id,),
            )
            step = cur.fetchone()
            if not step:
                return "no_action"

            canary_reqs = step["canary_requests"]
            canary_errs = step["canary_errors"]
            canary_p95 = step["canary_p95_ms"]
            error_rate = (canary_errs / canary_reqs * 100) if canary_reqs > 0 else 0

            now = datetime.utcnow().isoformat()

            # Check rollback conditions
            if canary_reqs >= config["min_requests_per_step"]:
                if error_rate > config["max_error_rate_pct"]:
                    return self._do_rollback(deployment_id, deploy, f"Error rate {error_rate:.1f}% > {config['max_error_rate_pct']}%", error_rate, canary_p95)
                if canary_p95 > config["max_latency_p95_ms"]:
                    return self._do_rollback(deployment_id, deploy, f"p95 latency {canary_p95:.0f}ms > {config['max_latency_p95_ms']}ms", error_rate, canary_p95)

            # Check promotion
            current_weight = deploy["canary_weight"]
            if current_weight >= config["promotion_threshold_pct"]:
                cur.execute(
                    "UPDATE deployments SET state = ?, canary_weight = 100, updated_at = ?, completed_at = ? WHERE deployment_id = ?",
                    (DeploymentState.PROMOTED.value, now, now, deployment_id),
                )
                cur.execute(
                    "INSERT INTO deployment_steps (deployment_id, step_number, canary_weight, action, timestamp) VALUES (?,?,?,?,?)",
                    (deployment_id, step["step_number"] + 1, 100, "promoted", now),
                )
                logger.info("Promoted canary %s — v%d is now production", deployment_id, deploy["canary_version"])
                return "promoted"

            # Ramp up
            if canary_reqs >= config["min_requests_per_step"]:
                new_weight = min(100, current_weight + config["weight_step_pct"])
                cur.execute(
                    "UPDATE deployments SET canary_weight = ?, updated_at = ? WHERE deployment_id = ?",
                    (new_weight, now, deployment_id),
                )
                cur.execute(
                    "INSERT INTO deployment_steps (deployment_id, step_number, canary_weight, canary_requests, canary_errors, canary_p95_ms, action, timestamp) VALUES (?,?,?,0,0,0,?,?)",
                    (deployment_id, step["step_number"] + 1, new_weight, "ramp_up", now),
                )
                logger.info("Ramped up %s: %d%% → %d%%", deployment_id, int(current_weight), int(new_weight))
                return "ramp_up"

            return "hold"

    def _do_rollback(self, deployment_id: str, deploy: sqlite3.Row, reason: str, error_rate: float, p95: float) -> str:
        now = datetime.utcnow().isoformat()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE deployments SET state = ?, canary_weight = 0, updated_at = ?, completed_at = ? WHERE deployment_id = ?",
                (DeploymentState.ROLLED_BACK.value, now, now, deployment_id),
            )
            cur.execute(
                "INSERT INTO rollback_log (deployment_id, reason, canary_version, rolled_back_to, error_rate_pct, p95_latency_ms, timestamp) VALUES (?,?,?,?,?,?,?)",
                (deployment_id, reason, deploy["canary_version"], deploy["baseline_version"], error_rate, p95, now),
            )
        logger.warning("ROLLBACK %s: %s", deployment_id, reason)
        return "rolled_back"

    # ── Status / History ──────────────────────────────────────
    def get_status(self, deployment_id: str) -> Optional[CanaryStatus]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,))
            d = cur.fetchone()
            if not d:
                return None

            cur.execute(
                "SELECT COUNT(*) as steps, SUM(canary_requests) as reqs, SUM(canary_errors) as errs FROM deployment_steps WHERE deployment_id = ?",
                (deployment_id,),
            )
            agg = cur.fetchone()

            cur.execute(
                "SELECT canary_p95_ms FROM deployment_steps WHERE deployment_id = ? ORDER BY step_number DESC LIMIT 1",
                (deployment_id,),
            )
            latest = cur.fetchone()

            total_reqs = agg["reqs"] or 0
            total_errs = agg["errs"] or 0
            started = datetime.fromisoformat(d["started_at"])
            elapsed = (datetime.utcnow() - started).total_seconds() / 60

            return CanaryStatus(
                deployment_id=deployment_id,
                model_name=d["model_name"],
                canary_version=d["canary_version"],
                baseline_version=d["baseline_version"],
                state=d["state"],
                canary_weight=d["canary_weight"],
                steps_completed=agg["steps"] or 0,
                total_canary_requests=total_reqs,
                total_canary_errors=total_errs,
                current_error_rate=round(total_errs / total_reqs * 100, 2) if total_reqs > 0 else 0,
                current_p95_ms=latest["canary_p95_ms"] if latest else 0,
                started_at=d["started_at"],
                elapsed_minutes=round(elapsed, 1),
            )

    def list_deployments(self, model_name: Optional[str] = None, limit: int = 20) -> list[dict]:
        with self._cursor() as cur:
            if model_name:
                cur.execute(
                    "SELECT * FROM deployments WHERE model_name = ? ORDER BY started_at DESC LIMIT ?",
                    (model_name, limit),
                )
            else:
                cur.execute("SELECT * FROM deployments ORDER BY started_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_rollback_history(self, limit: int = 20) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM rollback_log ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ── Seed demo data ────────────────────────────────────────
    def seed_sample_data(self):
        """Seed canary deployment history for demo."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM deployments")
            if cur.fetchone()["c"] > 0:
                return

        now = datetime.utcnow()

        # Deployment 1: Successful canary (v1→v2, promoted)
        dep1 = f"deploy-nvidia-lstm-forecast-v2-{(now - timedelta(days=3)).strftime('%Y%m%d%H%M%S')}"
        with self._cursor() as cur:
            started = (now - timedelta(days=3)).isoformat()
            completed = (now - timedelta(days=3, hours=-2)).isoformat()
            cur.execute(
                """INSERT INTO deployments (deployment_id, model_name, canary_version, baseline_version, state, canary_weight, config_json, started_at, updated_at, completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (dep1, "nvidia-lstm-forecast", 2, 1, "promoted", 100, json.dumps(DEFAULT_CANARY_CONFIG), started, completed, completed),
            )
            for i, w in enumerate([5, 15, 25, 35, 45, 55, 65, 75, 85, 100], 1):
                ts = (now - timedelta(days=3) + timedelta(minutes=i * 12)).isoformat()
                cur.execute(
                    "INSERT INTO deployment_steps (deployment_id, step_number, canary_weight, canary_requests, canary_errors, canary_p95_ms, baseline_requests, baseline_errors, baseline_p95_ms, health_ok, action, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (dep1, i, w, random.randint(20, 100), random.randint(0, 1), random.uniform(80, 200),
                     random.randint(50, 200), random.randint(0, 2), random.uniform(90, 180), 1,
                     "promoted" if w == 100 else "ramp_up", ts),
                )

        # Deployment 2: Rolled back (v3 canary failed)
        dep2 = f"deploy-nvidia-lstm-forecast-v3-{(now - timedelta(days=1)).strftime('%Y%m%d%H%M%S')}"
        with self._cursor() as cur:
            started = (now - timedelta(days=1)).isoformat()
            completed = (now - timedelta(days=1, hours=-0.5)).isoformat()
            cur.execute(
                """INSERT INTO deployments (deployment_id, model_name, canary_version, baseline_version, state, canary_weight, config_json, started_at, updated_at, completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (dep2, "nvidia-lstm-forecast", 3, 2, "rolled_back", 0, json.dumps(DEFAULT_CANARY_CONFIG), started, completed, completed),
            )
            for i, w in enumerate([5, 15, 25], 1):
                ts = (now - timedelta(days=1) + timedelta(minutes=i * 5)).isoformat()
                errs = random.randint(0, 1) if i < 3 else random.randint(5, 10)
                cur.execute(
                    "INSERT INTO deployment_steps (deployment_id, step_number, canary_weight, canary_requests, canary_errors, canary_p95_ms, baseline_requests, baseline_errors, baseline_p95_ms, health_ok, action, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (dep2, i, w, random.randint(15, 40), errs, random.uniform(100, 800 if i == 3 else 200),
                     random.randint(50, 100), random.randint(0, 1), random.uniform(90, 160), 0 if i == 3 else 1,
                     "rolled_back" if i == 3 else "ramp_up", ts),
                )
            cur.execute(
                "INSERT INTO rollback_log (deployment_id, reason, canary_version, rolled_back_to, error_rate_pct, p95_latency_ms, timestamp) VALUES (?,?,?,?,?,?,?)",
                (dep2, "Error rate 18.5% > 2.0% threshold", 3, 2, 18.5, 742.3, completed),
            )

        logger.info("Seeded canary deployment history (1 success, 1 rollback)")
