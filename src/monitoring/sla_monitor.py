"""SLA Monitor — uptime, latency, and availability tracking.

Tracks:
    - API uptime (periodic health checks)
    - Response latency percentiles (p50, p95, p99)
    - Error rate SLA
    - Model inference latency SLA
    - Availability windows

All data stored in SQLite for historical dashboarding.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sla_metrics.db"

# SLA thresholds
SLA_TARGETS = {
    "uptime_pct": 99.5,  # 99.5% uptime target
    "latency_p95_ms": 500,  # p95 latency < 500ms
    "latency_p99_ms": 2000,  # p99 latency < 2s
    "error_rate_pct": 1.0,  # < 1% error rate
    "inference_latency_ms": 200,  # model inference < 200ms
}

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS health_checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL DEFAULT '/health',
    status      TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    status_code INTEGER DEFAULT 200
);

CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms  REAL    NOT NULL,
    is_error    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sla_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    period_minutes      INTEGER NOT NULL,
    uptime_pct          REAL,
    total_checks        INTEGER,
    successful_checks   INTEGER,
    avg_latency_ms      REAL,
    p50_latency_ms      REAL,
    p95_latency_ms      REAL,
    p99_latency_ms      REAL,
    error_rate_pct      REAL,
    total_requests      INTEGER,
    error_requests      INTEGER,
    sla_met             INTEGER DEFAULT 1
);
"""


@dataclass
class SLAReport:
    """Current SLA compliance report."""

    uptime_pct: float = 100.0
    total_checks: int = 0
    successful_checks: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate_pct: float = 0.0
    total_requests: int = 0
    error_requests: int = 0
    sla_targets: dict = field(default_factory=lambda: SLA_TARGETS.copy())
    violations: list = field(default_factory=list)
    overall_sla_met: bool = True


class SLAMonitor:
    """Monitors and tracks SLA metrics."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "SLAMonitor":
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hc_ts ON health_checks(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_ts ON request_log(timestamp);")
        conn.commit()

    # ── Record events ─────────────────────────────────────────
    def record_health_check(self, status: str, latency_ms: float, status_code: int = 200):
        """Record a health check result."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO health_checks (timestamp, endpoint, status, latency_ms, status_code) VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), "/health", status, latency_ms, status_code),
            )

    def record_request(self, method: str, endpoint: str, status_code: int, latency_ms: float):
        """Record an API request for SLA tracking."""
        is_error = 1 if status_code >= 500 else 0
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO request_log
                   (timestamp, method, endpoint, status_code,
                    latency_ms, is_error)
                   VALUES (?,?,?,?,?,?)""",
                (datetime.utcnow().isoformat(), method, endpoint, status_code, latency_ms, is_error),
            )

    # ── Compute SLA ───────────────────────────────────────────
    def compute_sla(self, period_minutes: int = 60) -> SLAReport:
        """Compute SLA report for the given time window."""
        cutoff = (datetime.utcnow() - timedelta(minutes=period_minutes)).isoformat()
        report = SLAReport()

        with self._cursor() as cur:
            # Health check uptime
            cur.execute("SELECT COUNT(*) as total FROM health_checks WHERE timestamp >= ?", (cutoff,))
            report.total_checks = cur.fetchone()["total"]

            cur.execute(
                "SELECT COUNT(*) as ok FROM health_checks WHERE timestamp >= ? AND status = 'healthy'",
                (cutoff,),
            )
            report.successful_checks = cur.fetchone()["ok"]

            if report.total_checks > 0:
                report.uptime_pct = round(report.successful_checks / report.total_checks * 100, 2)
            else:
                report.uptime_pct = 100.0

            # Request latency percentiles
            cur.execute(
                "SELECT latency_ms FROM request_log WHERE timestamp >= ? ORDER BY latency_ms",
                (cutoff,),
            )
            latencies = [r["latency_ms"] for r in cur.fetchall()]
            report.total_requests = len(latencies)

            if latencies:
                import numpy as np

                arr = np.array(latencies)
                report.avg_latency_ms = round(float(np.mean(arr)), 1)
                report.p50_latency_ms = round(float(np.percentile(arr, 50)), 1)
                report.p95_latency_ms = round(float(np.percentile(arr, 95)), 1)
                report.p99_latency_ms = round(float(np.percentile(arr, 99)), 1)

            # Error rate
            cur.execute(
                "SELECT COUNT(*) as errs FROM request_log WHERE timestamp >= ? AND is_error = 1",
                (cutoff,),
            )
            report.error_requests = cur.fetchone()["errs"]
            if report.total_requests > 0:
                report.error_rate_pct = round(report.error_requests / report.total_requests * 100, 2)

        # Check violations
        report.violations = []
        if report.uptime_pct < SLA_TARGETS["uptime_pct"]:
            report.violations.append(f"Uptime {report.uptime_pct}% < target {SLA_TARGETS['uptime_pct']}%")
        if report.p95_latency_ms > SLA_TARGETS["latency_p95_ms"]:
            report.violations.append(
                f"p95 latency {report.p95_latency_ms}ms > target {SLA_TARGETS['latency_p95_ms']}ms"
            )
        if report.error_rate_pct > SLA_TARGETS["error_rate_pct"]:
            report.violations.append(f"Error rate {report.error_rate_pct}% > target {SLA_TARGETS['error_rate_pct']}%")

        report.overall_sla_met = len(report.violations) == 0
        return report

    def get_uptime_history(self, days: int = 7) -> list[dict]:
        """Get daily uptime history for charting."""
        result = []
        for i in range(days):
            day = datetime.utcnow() - timedelta(days=days - 1 - i)
            day_start = day.replace(hour=0, minute=0, second=0).isoformat()
            day_end = day.replace(hour=23, minute=59, second=59).isoformat()

            with self._cursor() as cur:
                cur.execute(
                    """SELECT COUNT(*) as total,
                       SUM(CASE WHEN status='healthy' THEN 1 ELSE 0 END) as ok
                       FROM health_checks
                       WHERE timestamp BETWEEN ? AND ?""",
                    (day_start, day_end),
                )
                row = cur.fetchone()
                total = row["total"]
                ok = row["ok"] or 0
                uptime = round(ok / total * 100, 1) if total > 0 else 100.0

                cur.execute(
                    """SELECT COUNT(*) as reqs, AVG(latency_ms) as avg_lat
                       FROM request_log
                       WHERE timestamp BETWEEN ? AND ?""",
                    (day_start, day_end),
                )
                req_row = cur.fetchone()

            result.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "uptime_pct": uptime,
                    "checks": total,
                    "requests": req_row["reqs"],
                    "avg_latency_ms": round(req_row["avg_lat"] or 0, 1),
                }
            )
        return result

    def seed_sample_data(self):
        """Seed realistic SLA data for demo."""
        import random

        random.seed(123)

        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM health_checks")
            if cur.fetchone()["c"] > 0:
                return

        now = datetime.utcnow()
        endpoints = ["/health", "/predict", "/data", "/agent/query", "/model/info", "/evaluation/explainability"]

        for i in range(1440):  # 24h of data, one per minute
            ts = (now - timedelta(minutes=1440 - i)).isoformat()
            import random

            # Health checks (every minute)
            healthy = random.random() > 0.005  # 99.5% uptime
            latency = random.gauss(15, 5) if healthy else random.gauss(5000, 1000)
            with self._cursor() as cur:
                cur.execute(
                    """INSERT INTO health_checks
                       (timestamp, endpoint, status,
                        latency_ms, status_code)
                       VALUES (?,?,?,?,?)""",
                    (ts, "/health", "healthy" if healthy else "unhealthy", max(1, latency), 200 if healthy else 503),
                )

            # Simulate ~3 requests per minute
            for _ in range(random.randint(1, 5)):
                ep = random.choice(endpoints)
                base_lat = {"predict": 150, "/agent/query": 2500, "/data": 30}.get(ep, 50)
                lat = max(1, random.gauss(base_lat, base_lat * 0.3))
                code = random.choices([200, 400, 500], weights=[97, 2, 1])[0]
                with self._cursor() as cur:
                    cur.execute(
                        """INSERT INTO request_log
                           (timestamp, method, endpoint,
                            status_code, latency_ms, is_error)
                           VALUES (?,?,?,?,?,?)""",
                        (ts, "GET" if ep != "/predict" else "POST", ep, code, lat, 1 if code >= 500 else 0),
                    )

        logger.info("Seeded 24h of SLA sample data")
