"""Structured log storage using SQLite.

Provides a SQLite-backed logging handler so every log record from
the application (FastAPI, Agent, Training, ETL, Monitoring, etc.)
is persisted with structured fields for querying and charting.
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "logs.db"

# ─── Schema ────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    level       TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    module      TEXT    NOT NULL DEFAULT '',
    func_name   TEXT    NOT NULL DEFAULT '',
    message     TEXT    NOT NULL,
    extra       TEXT    NOT NULL DEFAULT '{}',
    created_at  REAL    NOT NULL
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_logs_level     ON logs(level);",
    "CREATE INDEX IF NOT EXISTS idx_logs_source    ON logs(source);",
    "CREATE INDEX IF NOT EXISTS idx_logs_created   ON logs(created_at);",
]


# ─── Data classes ──────────────────────────────────────────────────
@dataclass
class LogEntry:
    id: int
    timestamp: str
    level: str
    source: str
    module: str
    func_name: str
    message: str
    extra: dict = field(default_factory=dict)


@dataclass
class LogStats:
    total: int = 0
    by_level: dict = field(default_factory=dict)
    by_source: dict = field(default_factory=dict)
    error_rate: float = 0.0
    warning_rate: float = 0.0
    logs_per_minute: float = 0.0


# ─── Database manager ─────────────────────────────────────────────
class LogDatabase:
    """Thread-safe SQLite log database."""

    _instance: Optional["LogDatabase"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "LogDatabase":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
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
        with self._cursor() as cur:
            cur.execute(CREATE_TABLE)
            for idx_sql in CREATE_INDEXES:
                cur.execute(idx_sql)

    # ── Write ───────────────────────────────────────────────────
    def insert(
        self,
        level: str,
        source: str,
        message: str,
        module: str = "",
        func_name: str = "",
        extra: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ):
        ts = timestamp or datetime.utcnow().isoformat(timespec="milliseconds")
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO logs (timestamp, level, source, module, func_name, message, extra, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, level, source, module, func_name, message, json.dumps(extra or {}), time.time()),
            )

    def insert_batch(self, records: list[tuple]):
        with self._cursor() as cur:
            cur.executemany(
                """INSERT INTO logs (timestamp, level, source, module, func_name, message, extra, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                records,
            )

    # ── Read ────────────────────────────────────────────────────
    def query(
        self,
        level: Optional[str] = None,
        source: Optional[str] = None,
        search: Optional[str] = None,
        since_minutes: int = 60,
        limit: int = 500,
        offset: int = 0,
    ) -> list[LogEntry]:
        clauses = ["1=1"]
        params: list[Any] = []

        if level:
            clauses.append("level = ?")
            params.append(level.upper())
        if source:
            clauses.append("source = ?")
            params.append(source)
        if since_minutes > 0:
            cutoff = (datetime.utcnow() - timedelta(minutes=since_minutes)).isoformat()
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        if search:
            clauses.append("message LIKE ?")
            params.append(f"%{search}%")

        params.extend([limit, offset])
        sql = f"""
            SELECT id, timestamp, level, source, module, func_name, message, extra
            FROM logs
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            LogEntry(
                id=r["id"],
                timestamp=r["timestamp"],
                level=r["level"],
                source=r["source"],
                module=r["module"],
                func_name=r["func_name"],
                message=r["message"],
                extra=json.loads(r["extra"]) if r["extra"] else {},
            )
            for r in rows
        ]

    def get_stats(self, since_minutes: int = 60) -> LogStats:
        cutoff = (datetime.utcnow() - timedelta(minutes=since_minutes)).isoformat()

        with self._cursor() as cur:
            # Total
            cur.execute("SELECT COUNT(*) AS c FROM logs WHERE timestamp >= ?", (cutoff,))
            total = cur.fetchone()["c"]

            # By level
            cur.execute(
                "SELECT level, COUNT(*) AS c FROM logs WHERE timestamp >= ? GROUP BY level ORDER BY c DESC",
                (cutoff,),
            )
            by_level = {r["level"]: r["c"] for r in cur.fetchall()}

            # By source
            cur.execute(
                "SELECT source, COUNT(*) AS c FROM logs WHERE timestamp >= ? GROUP BY source ORDER BY c DESC",
                (cutoff,),
            )
            by_source = {r["source"]: r["c"] for r in cur.fetchall()}

        error_count = by_level.get("ERROR", 0)
        warning_count = by_level.get("WARNING", 0)

        return LogStats(
            total=total,
            by_level=by_level,
            by_source=by_source,
            error_rate=round((error_count / total * 100) if total else 0, 1),
            warning_rate=round((warning_count / total * 100) if total else 0, 1),
            logs_per_minute=round(total / max(since_minutes, 1), 2),
        )

    def get_timeline(self, since_minutes: int = 60, bucket_minutes: int = 1) -> list[dict]:
        """Aggregate log counts into time buckets for charting."""
        cutoff = (datetime.utcnow() - timedelta(minutes=since_minutes)).isoformat()

        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    SUBSTR(timestamp, 1, 16) AS bucket,
                    level,
                    COUNT(*) AS c
                FROM logs
                WHERE timestamp >= ?
                GROUP BY bucket, level
                ORDER BY bucket
                """,
                (cutoff,),
            )
            rows = cur.fetchall()

        # Pivot into {bucket: {INFO: n, ERROR: n, ...}}
        buckets: dict[str, dict] = {}
        for r in rows:
            b = r["bucket"]
            if b not in buckets:
                buckets[b] = {"time": b, "INFO": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}
            level = r["level"]
            if level in buckets[b]:
                buckets[b][level] = r["c"]

        return list(buckets.values())

    def get_sources(self) -> list[str]:
        with self._cursor() as cur:
            cur.execute("SELECT DISTINCT source FROM logs ORDER BY source")
            return [r["source"] for r in cur.fetchall()]

    def cleanup(self, keep_hours: int = 72):
        """Delete logs older than keep_hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=keep_hours)).isoformat()
        with self._cursor() as cur:
            cur.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
        if deleted:
            logger.info("Cleaned up %d old log entries", deleted)
        return deleted


# ─── Logging Handler ───────────────────────────────────────────────
class SQLiteLogHandler(logging.Handler):
    """Python logging handler that writes records to the log database."""

    # Map Python logger names to friendly source labels
    SOURCE_MAP = {
        "src.api": "fastapi",
        "src.agent": "agent",
        "src.training": "training",
        "src.data": "data",
        "src.etl": "etl",
        "src.monitoring": "monitoring",
        "src.models": "model",
        "src.explainability": "explainability",
        "src.prediction": "prediction",
        "src.security": "security",
        "src.dashboard": "dashboard",
        "uvicorn": "uvicorn",
        "fastapi": "fastapi",
    }

    def __init__(self, db: Optional[LogDatabase] = None, min_level: int = logging.DEBUG):
        super().__init__(level=min_level)
        self.db = db or LogDatabase.get_instance()
        self._buffer: list[tuple] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size = 20
        self._last_flush = time.time()

    def _resolve_source(self, name: str) -> str:
        for prefix, label in self.SOURCE_MAP.items():
            if name.startswith(prefix):
                return label
        return "app"

    def emit(self, record: logging.LogRecord):
        try:
            ts = datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds")
            source = self._resolve_source(record.name)
            extra = {}
            if record.exc_info and record.exc_info[1]:
                extra["exception"] = str(record.exc_info[1])
            if hasattr(record, "request_id"):
                extra["request_id"] = record.request_id

            entry = (
                ts,
                record.levelname,
                source,
                record.module,
                record.funcName or "",
                record.getMessage(),
                json.dumps(extra),
                time.time(),
            )

            with self._buffer_lock:
                self._buffer.append(entry)
                if len(self._buffer) >= self._buffer_size or (time.time() - self._last_flush) > 2:
                    self._flush()
        except Exception:
            self.handleError(record)

    def _flush(self):
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.time()
        try:
            self.db.insert_batch(batch)
        except Exception as exc:
            # Fallback — avoid infinite recursion
            import sys
            print(f"[SQLiteLogHandler] flush error: {exc}", file=sys.stderr)

    def flush(self):
        with self._buffer_lock:
            self._flush()

    def close(self):
        self.flush()
        super().close()


# ─── Setup helper ──────────────────────────────────────────────────
_handler_installed = False


def install_log_handler(level: int = logging.INFO):
    """Install the SQLite log handler on the root logger (idempotent)."""
    global _handler_installed
    if _handler_installed:
        return
    _handler_installed = True

    db = LogDatabase.get_instance()
    handler = SQLiteLogHandler(db=db, min_level=level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)

    # Also capture uvicorn access logs
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvi = logging.getLogger(name)
        uvi.addHandler(handler)

    logger.info("SQLite log handler installed — logs will be stored in %s", db.db_path)


def seed_sample_logs():
    """Insert sample logs so charts are never empty."""
    import random

    db = LogDatabase.get_instance()
    sources = ["fastapi", "agent", "training", "etl", "monitoring", "model", "security"]
    levels = ["INFO", "INFO", "INFO", "INFO", "WARNING", "ERROR", "DEBUG"]
    messages = {
        "fastapi": [
            "POST /predict 200 OK (123ms)",
            "GET /health 200 OK (2ms)",
            "POST /agent/query 200 OK (3421ms)",
            "GET /data/live 200 OK (45ms)",
            "POST /monitoring/drift 200 OK (890ms)",
            "GET /model/info 200 OK (5ms)",
            "Request validation error on /predict",
        ],
        "agent": [
            "ReAct agent initialized with 5 tools",
            "Agent query: 'What is the NVIDIA stock trend?'",
            "Tool called: query_stock_data",
            "Tool called: rag_search",
            "Agent completed in 2 iterations",
            "Guardrail check passed for input",
            "Output sanitized — disclaimer added",
        ],
        "training": [
            "Epoch 1/50 — loss: 0.0234 val_loss: 0.0312",
            "Epoch 25/50 — loss: 0.0098 val_loss: 0.0145",
            "Epoch 50/50 — loss: 0.0067 val_loss: 0.0112",
            "Early stopping triggered at epoch 42",
            "Best model saved: RMSE=0.1216",
            "HPO trial 3: hidden=128, layers=2, lr=0.001",
            "Training completed — best RMSE: 0.1102",
        ],
        "etl": [
            "Extracting NVIDIA stock data from Yahoo Finance",
            "Downloaded 1260 rows (2021-01-01 to 2026-03-27)",
            "Feature engineering: added SMA_20, SMA_50, RSI_14",
            "Data saved to data/processed/nvidia_processed.csv",
            "Scaler fitted and saved to models/scaler.pkl",
            "Data validation passed — no nulls detected",
        ],
        "monitoring": [
            "Drift check: KS statistic=0.0523 (p=0.82) — no drift",
            "Drift check: PSI=0.032 — below threshold",
            "Champion RMSE: 0.1216 | Challenger RMSE: 0.1389",
            "Performance alert: prediction error > 15%",
            "Model performance stable — no retraining needed",
        ],
        "model": [
            "NvidiaLSTM loaded: input=5, hidden=128, layers=2",
            "Model parameters: 198,789 trainable",
            "Inference: batch of 1, latency 12ms",
            "Model checkpoint saved",
        ],
        "security": [
            "Input guardrail: passed validation",
            "Output guardrail: PII check — clean",
            "Rate limit: 45/100 requests this minute",
            "Prompt injection attempt blocked",
        ],
    }

    records = []
    now = datetime.utcnow()
    for i in range(200):
        minutes_ago = random.randint(0, 120)
        ts = (now - timedelta(minutes=minutes_ago, seconds=random.randint(0, 59))).isoformat(
            timespec="milliseconds"
        )
        source = random.choice(sources)
        level = random.choice(levels)
        msg_list = messages.get(source, ["Log entry"])
        msg = random.choice(msg_list)

        # Make errors more realistic
        if level == "ERROR":
            msg = f"ERROR — {msg}" if "error" not in msg.lower() else msg

        records.append((ts, level, source, source, "", msg, "{}", time.time() - minutes_ago * 60))

    db.insert_batch(records)
    logger.info("Seeded %d sample log entries", len(records))
